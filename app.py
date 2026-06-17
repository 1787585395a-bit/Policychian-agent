from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from time import perf_counter
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from policychain.mcp import StdioMCPInvoker, cache_mcp_invoker
from policychain.llm import LLMConfigurationError
from policychain.paths import resolve_default_db_path
from scripts.run_research import run_research


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_INPUT = """生成式人工智能服务管理办法

第一条 为促进生成式人工智能健康发展和规范应用，维护国家安全和社会公共利益，制定本办法。
第二条 生成式人工智能服务提供者应当依法承担网络信息内容生产者责任，履行网络信息安全义务。
第三条 服务提供者应当开展算法模型安全评估，提升训练数据质量，采取有效措施防范违法和不良信息生成传播。
第四条 主管部门应当加强监督管理，推动行业组织建立服务能力评估机制。"""
EXAMPLE_QUERIES = (
    "https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm",
)

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _resolve_host(env: dict[str, str] | None = None) -> str:
    values = env if env is not None else os.environ
    return values.get("POLICYCHAIN_HOST", "127.0.0.1")


def _resolve_port(env: dict[str, str] | None = None) -> int:
    values = env if env is not None else os.environ
    return int(values.get("PORT") or values.get("POLICYCHAIN_PORT", "8000"))


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sync_jobs_enabled(env: dict[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _truthy_env(values.get("POLICYCHAIN_SYNC_JOBS")) or _truthy_env(values.get("VERCEL"))


HOST = _resolve_host()
PORT = _resolve_port()


def run_query(
    query: str,
    db_path: str | Path | None = None,
    use_llm: bool = False,
    use_mcp: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    effective_use_llm = use_llm
    mcp_invoker = None
    runtime_notes: list[str] = []
    if use_mcp:
        try:
            mcp_invoker = _build_mcp_invoker()
            if progress_callback:
                progress_callback(2, "MCP 初始化", "MCP 外部工具初始化成功，已启用 Open-WebSearch/CNFinancial 工具通道")
        except Exception as exc:
            message = f"MCP 外部工具初始化失败，已回退到本地流程：{exc}"
            runtime_notes.append(message)
            if progress_callback:
                progress_callback(2, "MCP 初始化", message)
    try:
        try:
            report = run_research(
                query=query.strip() or DEFAULT_POLICY_INPUT,
                db_path=db_path or resolve_default_db_path(),
                ensure_sample_db=True,
                rebuild_sample_db=False,
                use_llm=effective_use_llm,
                mcp_invoker=mcp_invoker,
                progress_callback=progress_callback,
            )
        except LLMConfigurationError as exc:
            if not effective_use_llm:
                raise
            effective_use_llm = False
            message = f"模型分析初始化失败，已回退到确定性流程：{exc}"
            runtime_notes.append(message)
            if progress_callback:
                progress_callback(3, "模型初始化", message)
            report = run_research(
                query=query.strip() or DEFAULT_POLICY_INPUT,
                db_path=db_path or resolve_default_db_path(),
                ensure_sample_db=True,
                rebuild_sample_db=False,
                use_llm=False,
                mcp_invoker=mcp_invoker,
                progress_callback=progress_callback,
            )
    finally:
        closer = getattr(mcp_invoker, "close", None)
        if callable(closer):
            closer()
    if runtime_notes:
        notes = "\n".join(f"- {escape_note}" for escape_note in runtime_notes)
        report = f"{report}\n\n## 运行环境提示\n\n{notes}"
    return {
        "query": query.strip() or DEFAULT_POLICY_INPUT,
        "report": report,
        "use_llm": effective_use_llm,
        "use_mcp": bool(mcp_invoker),
        "elapsed_seconds": round(perf_counter() - started, 2),
    }


def render_page(
    query: str = DEFAULT_POLICY_INPUT,
    report: str = "",
    error: str = "",
    elapsed_seconds: float | None = None,
    use_llm: bool = True,
    use_mcp: bool = True,
) -> bytes:
    examples = "\n".join(
        f'<button class="quick" type="button" data-query="{escape(item)}">{escape(item)}</button>'
        for item in EXAMPLE_QUERIES
    )
    report_html = (
        _markdown_to_html(report)
        if report
        else '<div class="empty">等待分析结果。运行后将在这里显示政策分析、相似政策对比、行业影响和公司业务匹配报告。</div>'
    )
    error_html = f'<div class="error" id="error-box">{escape(error)}</div>' if error else '<div class="error hidden" id="error-box"></div>'

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PolicyChain</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #ffffff;
      --soft: #f7f7f8;
      --text: #101010;
      --muted: #6e6e73;
      --line: #dedede;
      --line-strong: #c7c7c7;
      --accent: #111111;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    html {{ max-width: 100%; overflow-x: hidden; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
      overflow-x: hidden;
    }}
    button, textarea, input {{ font: inherit; }}
    header {{
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      position: sticky;
      top: 0;
      z-index: 5;
      backdrop-filter: blur(12px);
    }}
    .topbar {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 15px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .brand {{ font-size: 16px; font-weight: 650; line-height: 1; white-space: nowrap; }}
    main {{
      max-width: 1060px;
      margin: 0 auto;
      padding: 58px 24px 64px;
    }}
    .hero {{
      display: grid;
      justify-items: center;
      text-align: center;
      padding: 24px 0 28px;
    }}
    .hero h1 {{
      margin: 0;
      width: 100%;
      max-width: 960px;
      font-size: clamp(36px, 5.5vw, 64px);
      line-height: 1.08;
      font-weight: 650;
      letter-spacing: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .hero p {{
      margin: 20px 0 0;
      width: 100%;
      max-width: 700px;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.55;
      overflow-wrap: anywhere;
    }}
    .composer {{
      width: 100%;
      max-width: 860px;
      margin: 34px auto 0;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: #fff;
      text-align: left;
      overflow: hidden;
    }}
    .composer label[for="query"] {{
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
      white-space: nowrap;
    }}
    textarea {{
      display: block;
      width: 100%;
      min-height: 180px;
      resize: vertical;
      border: 0;
      padding: 20px 20px 14px;
      color: var(--text);
      background: #fff;
      font-size: 16px;
      line-height: 1.58;
      outline: none;
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: pre-wrap;
      overflow-x: hidden;
    }}
    textarea::placeholder {{ color: #8a8a8e; }}
    .composer-footer {{
      border-top: 1px solid var(--line);
      padding: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #fff;
    }}
    .run-button {{
      flex: 0 0 auto;
      min-width: 112px;
      border: 1px solid #111;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-size: 14px;
      font-weight: 600;
      line-height: 1;
      padding: 12px 18px;
      cursor: pointer;
      margin: 0 auto;
    }}
    .run-button:hover {{ background: #2b2b2b; }}
    .run-button:disabled {{ cursor: wait; background: #4a4a4a; border-color: #4a4a4a; }}
    .examples {{
      width: 100%;
      max-width: 860px;
      margin-top: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .quick {{
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      color: var(--text);
      font-size: 13px;
      line-height: 1.2;
      padding: 9px 11px;
      cursor: pointer;
      overflow-wrap: anywhere;
      word-break: break-all;
      text-align: left;
      white-space: normal;
      min-width: 0;
    }}
    .quick:hover {{ border-color: var(--line-strong); background: #fff; }}
    .status-line, .notice {{
      width: 100%;
      max-width: 860px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .progress-panel {{
      width: 100%;
      max-width: 860px;
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 16px;
      text-align: left;
    }}
    .progress-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .progress-track {{
      width: 100%;
      height: 8px;
      border-radius: 8px;
      background: var(--soft);
      overflow: hidden;
      border: 1px solid var(--line);
    }}
    .progress-bar {{
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width 220ms ease;
    }}
    .log-window-head {{
      margin-top: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .log-title {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      min-width: 0;
    }}
    .log-title strong {{ font-size: 14px; font-weight: 650; color: var(--text); }}
    .log-meta {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .copy-log {{
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      min-height: 34px;
      padding: 0 12px;
      font-size: 13px;
      cursor: pointer;
    }}
    .copy-log:disabled {{
      color: var(--muted);
      border-color: var(--line);
      background: var(--soft);
      cursor: not-allowed;
    }}
    .log-panel {{
      margin-top: 12px;
      height: 220px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px 12px;
      color: #2b2b2b;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 12px;
      line-height: 1.55;
    }}
    .log-item {{ padding: 4px 0; border-bottom: 1px solid #f1f1f1; white-space: pre-wrap; }}
    .log-item:last-child {{ border-bottom: 0; }}
    .result-wrap {{ margin-top: 32px; }}
    .error {{
      margin: 0 0 14px;
      padding: 14px 16px;
      color: var(--danger);
      border: 1px solid #fecdca;
      background: #fffbfa;
      border-radius: 8px;
      line-height: 1.55;
    }}
    .hidden {{ display: none; }}
    .report {{
      border-top: 1px solid var(--line);
      padding: 28px 0 0;
      line-height: 1.72;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .report h1 {{ margin: 0 0 18px; font-size: 30px; line-height: 1.2; font-weight: 650; }}
    .report h2 {{
      margin: 30px 0 12px;
      padding-top: 22px;
      border-top: 1px solid var(--line);
      font-size: 22px;
      line-height: 1.35;
      font-weight: 650;
    }}
    .report h3 {{ margin: 22px 0 10px; font-size: 18px; line-height: 1.45; font-weight: 650; }}
    .report h4 {{ margin: 20px 0 8px; font-size: 16px; line-height: 1.45; font-weight: 650; }}
    .report h5 {{ margin: 16px 0 8px; font-size: 15px; line-height: 1.45; font-weight: 650; color: #2b2b2b; }}
    .report p {{ margin: 10px 0; color: #1f1f1f; }}
    .report ul, .report ol {{ margin: 8px 0 14px; padding-left: 24px; }}
    .report li {{ margin: 6px 0; padding-left: 2px; }}
    .report strong {{ font-weight: 650; }}
    .empty {{ padding: 26px 0; color: var(--muted); font-size: 15px; line-height: 1.6; }}
    @media (max-width: 720px) {{
      .topbar {{ padding: 14px 16px; }}
      main {{ padding: 34px 16px 48px; }}
      .hero {{ padding-top: 14px; justify-items: start; text-align: left; }}
      .hero h1 {{ max-width: 100%; font-size: 30px; line-height: 1.14; word-break: break-all; }}
      .hero p, .composer, .examples, .status-line, .notice, .progress-panel {{ max-width: 100%; }}
      .examples {{ justify-content: flex-start; }}
      .quick {{ width: 100%; }}
      .composer-footer {{ align-items: stretch; }}
      .run-button {{ width: 100%; min-width: 0; }}
      .log-window-head {{ align-items: stretch; flex-direction: column; }}
      .log-meta {{ white-space: normal; }}
      .copy-log {{ width: 100%; }}
      .report h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <header>
    <nav class="topbar" aria-label="PolicyChain">
      <div class="brand">PolicyChain</div>
    </nav>
  </header>
  <main>
    <section class="hero">
      <h1>粘贴政策链接或政策正文</h1>
      <p>PolicyChain 会先读取你输入的政策内容，再检索本地知识库中的相似政策用于对比，随后完成政策分析、行业影响和公司业务匹配。</p>
      <form class="composer" method="post" action="/" id="research-form">
        <label for="query">政策链接或政策正文</label>
        <textarea id="query" name="query" wrap="soft" placeholder="粘贴 http/https 政策链接，或直接粘贴政策正文。">{escape(query)}</textarea>
        <div class="composer-footer">
          <button class="run-button" type="submit" data-idle-label="运行分析" data-loading-label="分析中">运行分析</button>
        </div>
      </form>
      <div class="examples" aria-label="快捷输入">
        {examples}
      </div>
      <div class="status-line" id="status-line" role="status" aria-live="polite">{_elapsed_text(elapsed_seconds)}</div>
      <div class="notice">研究辅助，不构成投资建议。默认启用模型分析与外部证据工具；不可用时会在日志和报告中说明。</div>
      <div class="progress-panel" aria-label="运行进度">
        <div class="progress-head">
          <span id="stage-label">等待开始</span>
          <span id="progress-label">0%</span>
        </div>
        <div class="progress-track"><div class="progress-bar" id="progress-bar"></div></div>
        <div class="log-window" aria-label="运行日志窗口">
          <div class="log-window-head">
            <div class="log-title">
              <strong>运行日志</strong>
              <span class="log-meta" id="log-meta">Job: - · 0 条</span>
            </div>
            <button class="copy-log" id="copy-log" type="button" disabled>复制日志</button>
          </div>
          <div class="log-panel" id="log-panel"><div class="log-item">等待提交政策链接或正文。</div></div>
        </div>
      </div>
    </section>
    <section class="result-wrap" aria-label="分析结果">
      {error_html}
      <article class="report" id="report">
        {report_html}
      </article>
    </section>
  </main>
  <script>
    const form = document.getElementById("research-form");
    const queryInput = document.getElementById("query");
    const runButton = form.querySelector(".run-button");
    const statusLine = document.getElementById("status-line");
    const stageLabel = document.getElementById("stage-label");
    const progressLabel = document.getElementById("progress-label");
    const progressBar = document.getElementById("progress-bar");
    const logPanel = document.getElementById("log-panel");
    const logMeta = document.getElementById("log-meta");
    const copyLogButton = document.getElementById("copy-log");
    const report = document.getElementById("report");
    const errorBox = document.getElementById("error-box");
    let pollTimer = null;
    let lastStatusPayload = {{status: "idle", progress: 0, stage: "等待开始", job_id: "", logs: []}};

    document.querySelectorAll(".quick").forEach((button) => {{
      button.addEventListener("click", () => {{
        queryInput.value = button.dataset.query || "";
        queryInput.focus();
      }});
    }});

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      clearInterval(pollTimer);
      runButton.disabled = true;
      runButton.textContent = runButton.dataset.loadingLabel;
      report.innerHTML = '<div class="empty">正在分析，请等待结果。</div>';
      errorBox.classList.add("hidden");
      errorBox.textContent = "";
      lastStatusPayload = {{status: "pending", progress: 0, stage: "排队中", job_id: "", logs: []}};
      copyLogButton.disabled = true;
      copyLogButton.textContent = "复制日志";
      renderStatus({{status: "pending", progress: 0, stage: "排队中", logs: [{{message: "任务已提交。"}}]}});
      try {{
        const response = await fetch("/api/research", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{
            query: queryInput.value,
            use_llm: true,
            use_mcp: true
          }})
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "任务创建失败");
        pollTimer = setInterval(() => pollStatus(payload.job_id), 1000);
        await pollStatus(payload.job_id);
      }} catch (error) {{
        showError(error.message || String(error));
        runButton.disabled = false;
        runButton.textContent = runButton.dataset.idleLabel;
      }}
    }});

    async function pollStatus(jobId) {{
      const response = await fetch(`/api/research-status?job_id=${{encodeURIComponent(jobId)}}`);
      const payload = await response.json();
      if (!response.ok) {{
        clearInterval(pollTimer);
        showError(payload.error || "状态读取失败");
        runButton.disabled = false;
        runButton.textContent = runButton.dataset.idleLabel;
        return;
      }}
      renderStatus(payload);
      if (payload.status === "done") {{
        clearInterval(pollTimer);
        report.innerHTML = payload.report_html || '<div class="empty">报告为空。</div>';
        runButton.disabled = false;
        runButton.textContent = runButton.dataset.idleLabel;
      }}
      if (payload.status === "error") {{
        clearInterval(pollTimer);
        showError(payload.error || "分析失败");
        runButton.disabled = false;
        runButton.textContent = runButton.dataset.idleLabel;
      }}
    }}

    function renderStatus(payload) {{
      lastStatusPayload = payload || lastStatusPayload;
      const progress = Math.max(0, Math.min(100, Number(payload.progress || 0)));
      stageLabel.textContent = payload.stage || "运行中";
      progressLabel.textContent = `${{progress}}%`;
      progressBar.style.width = `${{progress}}%`;
      statusLine.textContent = `当前阶段：${{payload.stage || "运行中"}}`;
      const logs = payload.logs || [];
      const jobId = payload.job_id || "-";
      logMeta.textContent = `Job: ${{jobId}} · ${{logs.length}} 条`;
      logPanel.innerHTML = logs.length
        ? logs.map((item) => `<div class="log-item">${{escapeHtml(formatLogLine(item))}}</div>`).join("")
        : '<div class="log-item">暂无日志。</div>';
      logPanel.scrollTop = logPanel.scrollHeight;
      copyLogButton.disabled = !(payload.status === "done" || payload.status === "error");
      if (copyLogButton.disabled) copyLogButton.textContent = "复制日志";
    }}

    copyLogButton.addEventListener("click", async () => {{
      const text = formatLogsForCopy(lastStatusPayload);
      try {{
        if (navigator.clipboard && window.isSecureContext) {{
          await navigator.clipboard.writeText(text);
        }} else {{
          fallbackCopyText(text);
        }}
        copyLogButton.textContent = "已复制";
        setTimeout(() => {{ copyLogButton.textContent = "复制日志"; }}, 1400);
      }} catch (error) {{
        fallbackCopyText(text);
        copyLogButton.textContent = "已复制";
        setTimeout(() => {{ copyLogButton.textContent = "复制日志"; }}, 1400);
      }}
    }});

    function formatLogLine(item) {{
      const time = formatLogTime(item.time || "");
      const progress = Number.isFinite(Number(item.progress)) ? `${{Number(item.progress)}}%` : "-";
      const stage = item.stage || "";
      const message = item.message || "";
      return `[${{time}}] [${{progress}}] ${{stage}} - ${{message}}`;
    }}

    function formatLogsForCopy(payload) {{
      const logs = payload.logs || [];
      const lines = [
        `PolicyChain Job: ${{payload.job_id || "-"}}`,
        `Status: ${{payload.status || "-"}}`,
        `Final stage: ${{payload.stage || "-"}}`,
        ""
      ];
      logs.forEach((item) => lines.push(formatLogLine(item)));
      if (payload.error) {{
        lines.push("");
        lines.push(`Error: ${{payload.error}}`);
      }}
      return lines.join("\\n");
    }}

    function formatLogTime(value) {{
      if (!value) return "-";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleTimeString("zh-CN", {{hour12: false}});
    }}

    function fallbackCopyText(text) {{
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }}

    function showError(message) {{
      errorBox.textContent = message;
      errorBox.classList.remove("hidden");
      statusLine.textContent = "分析失败";
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));
    }}
  </script>
</body>
</html>"""
    return html.encode("utf-8")


class PolicyChainRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json(_health_payload())
            return
        if parsed.path == "/api/research-status":
            params = parse_qs(parsed.query)
            job_id = (params.get("job_id") or [""])[0]
            self._send_json(_job_view(job_id))
            return
        self._send_html(render_page())

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/research":
            payload = self._read_payload()
            query = str(payload.get("query") or DEFAULT_POLICY_INPUT)
            use_llm = _payload_bool(payload, "use_llm", default=True)
            use_mcp = _payload_bool(payload, "use_mcp", default=True)
            job_id = _create_job(query=query, use_llm=use_llm, use_mcp=use_mcp)
            self._send_json({"job_id": job_id, "status": "pending"}, status=202)
            return

        payload = self._read_payload()
        query = str(payload.get("query") or DEFAULT_POLICY_INPUT)
        use_llm = _payload_bool(payload, "use_llm", default=True)
        use_mcp = _payload_bool(payload, "use_mcp", default=True)
        try:
            result = run_query(query, use_llm=use_llm, use_mcp=use_mcp)
            html = render_page(
                query=result["query"],
                report=result["report"],
                elapsed_seconds=result["elapsed_seconds"],
                use_llm=result["use_llm"],
                use_mcp=result["use_mcp"],
            )
        except Exception as exc:
            html = render_page(query=query, error=str(exc), use_llm=use_llm, use_mcp=use_mcp)
        self._send_html(html)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return json.loads(body.decode("utf-8") or "{}")
        form = parse_qs(body.decode("utf-8", errors="replace"))
        return {
            "query": (form.get("query") or [DEFAULT_POLICY_INPUT])[0],
            "use_llm": (form.get("use_llm") or ["1"])[0] != "0",
            "use_mcp": (form.get("use_mcp") or ["1"])[0] != "0",
        }

    def _send_html(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _create_job(query: str, use_llm: bool, use_mcp: bool) -> str:
    job_id = uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "stage": "排队中",
            "logs": [_log_event("排队中", "任务已创建", 0)],
            "report": "",
            "error": "",
            "query": query,
            "use_llm": use_llm,
            "use_mcp": use_mcp,
            "elapsed_seconds": None,
        }
    if _sync_jobs_enabled():
        _run_job(job_id)
    else:
        thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        thread.start()
    return job_id


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        query = str(job["query"])
        use_llm = bool(job["use_llm"])
        use_mcp = bool(job["use_mcp"])
    _update_job(job_id, status="running", progress=1, stage="启动任务", message="任务开始执行")
    try:
        result = run_query(
            query,
            use_llm=use_llm,
            use_mcp=use_mcp,
            progress_callback=lambda progress, stage, message: _update_job(
                job_id,
                status="running",
                progress=progress,
                stage=stage,
                message=message,
            ),
        )
        _update_job(
            job_id,
            status="done",
            progress=100,
            stage="完成",
            report=result["report"],
            elapsed_seconds=result["elapsed_seconds"],
        )
    except Exception as exc:
        _update_job(job_id, status="error", stage="错误", message=str(exc), error=str(exc))


def _update_job(
    job_id: str,
    status: str | None = None,
    progress: int | None = None,
    stage: str | None = None,
    message: str | None = None,
    report: str | None = None,
    error: str | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if stage is not None:
            job["stage"] = stage
        if report is not None:
            job["report"] = report
        if error is not None:
            job["error"] = error
        if elapsed_seconds is not None:
            job["elapsed_seconds"] = elapsed_seconds
        if message:
            job["logs"].append(_log_event(stage or str(job.get("stage") or ""), message, int(job.get("progress") or 0)))


def _job_view(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id) or {})
    if not job:
        return {"status": "error", "error": "任务不存在", "progress": 0, "stage": "错误", "logs": []}
    report = str(job.get("report") or "")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "stage": job.get("stage"),
        "logs": job.get("logs") or [],
        "report": report,
        "report_html": _markdown_to_html(report) if report else "",
        "error": job.get("error") or "",
        "elapsed_seconds": job.get("elapsed_seconds"),
    }


def _log_event(stage: str, message: str, progress: int) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "message": message,
        "progress": progress,
    }


def _elapsed_text(elapsed_seconds: float | None) -> str:
    if elapsed_seconds is None:
        return "默认启用模型分析与外部证据工具。本地政策库仅用于检索相似政策。"
    return f"分析完成，用时：{elapsed_seconds:.2f} 秒。"


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "policychain",
        "time": datetime.now(timezone.utc).isoformat(),
    }


def _build_mcp_invoker():
    config_path = os.environ.get("POLICYCHAIN_MCP_CONFIG", ".mcp.local.json")
    timeout_seconds = float(os.environ.get("POLICYCHAIN_MCP_TIMEOUT", "60"))
    stdio_invoker = StdioMCPInvoker.from_config_file(
        config_path,
        timeout_seconds=timeout_seconds,
        project_root=PROJECT_ROOT,
    )
    return cache_mcp_invoker(stdio_invoker)


def _markdown_to_html(markdown: str) -> str:
    html: list[str] = []
    paragraph: list[str] = []
    list_type: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            html.append(f"<p>{_render_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            html.append(f"</{list_type}>")
            list_type = None

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            close_list()
            continue

        heading = re.match(r"^(#{1,5})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            html.append(f"<h{level}>{_render_inline(heading.group(2))}</h{level}>")
            continue

        unordered = re.match(r"^[-*]\s+(.+)$", line)
        if unordered:
            flush_paragraph()
            if list_type != "ul":
                close_list()
                html.append("<ul>")
                list_type = "ul"
            html.append(f"<li>{_render_inline(unordered.group(1))}</li>")
            continue

        ordered = re.match(r"^\d+[.)]\s+(.+)$", line)
        if ordered:
            flush_paragraph()
            if list_type != "ol":
                close_list()
                html.append("<ol>")
                list_type = "ol"
            html.append(f"<li>{_render_inline(ordered.group(1))}</li>")
            continue

        close_list()
        paragraph.append(line)

    flush_paragraph()
    close_list()
    return "\n".join(html)


def _render_inline(text: str) -> str:
    escaped = escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), PolicyChainRequestHandler)
    print(f"PolicyChain running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
