# PolicyChain 中文说明

[English](README.md) | 中文

最新真实示例：[《生成式人工智能服务管理暂行办法》政策研究报告](docs/analysis_result_example.zh-CN.md)。该报告来自运行 `run-20260806T110626-e72098de7073`，覆盖 5 条行业影响路径和 12 条经过审核的公司—路径匹配；网页中的“示例报告”读取同一份文件。

PolicyChain 是一个面向政策研究的 Agentic RAG 系统。核心流程是：用户输入政策链接或政策正文，系统读取并校验政策内容，进行政策分析、相似政策对比、实施路径分析、行业影响分析，并生成 A 股公司业务匹配研究说明。

本项目不是股票推荐或投资建议系统。报告只能用于政策研究辅助，不得输出买入、卖出、目标价、确定性收益判断或推荐股票。

## 在线试用

当前 Hugging Face Spaces 网站：

```text
https://zengliao-policychain-agent.hf.space
```

健康检查地址：

```text
https://zengliao-policychain-agent.hf.space/healthz
```

在线版本默认使用 DeepSeek 完成模型分析；必须在 Hugging Face Space 的 Secrets 中配置 `DEEPSEEK_API_KEY`。由于 Hugging Face 免费 CPU 对 stdio MCP 调用较慢，云端默认关闭 MCP。确定性流程仅在 DeepSeek 不可用时作为带日志记录的回退。

## 用户使用流程

1. 打开网页。
2. 粘贴政策链接，或直接粘贴政策正文。
3. 点击“运行分析”。
4. 查看进度条和运行日志。
5. 等待报告生成。
6. 在终态查看 Run ID、实际运行模式、各 Agent 状态和 fallback 情况；成功或失败任务都可下载脱敏运行日志。

输入应是政策链接或较完整的政策正文，不是普通问答问题。若链接无法公开访问、需要登录、验证码或页面正文过短，请改为粘贴政策正文。

## 报告内容

报告通常包括：

- 政策核心内容、发布主体、政策目标和政策力度。
- 本地知识库中相似政策、历史政策或可比政策的对比。
- 政策措施到实施行为、产业链环节和行业经营变量的传导路径。
- 可能受到直接、间接或潜在影响的行业。
- A 股公司业务匹配清单或“暂未形成可靠匹配”的原因。
- 参考资料、工具依据和不确定性说明。

公司部分只表示业务匹配研究清单，不构成投资建议。

## 公司匹配流程

Company Matcher 默认采用 Web-first 流程：

```text
DeepSeek + Web Search 按 impact_id 发现具体公司
→ CNFinancial 按股票代码、证券简称和法定名称精确核验
→ 合并 Web 与主营业务证据
→ DeepSeek 评价
→ 固定规则复核
→ 每条路径最多 3 家进入报告
```

每条路径最多生成 6 个未验证 Seed，候选评估池默认保留 5 家。只要公司身份有效、业务资料可追溯且没有明确冲突，弱相关或间接相关候选会以低置信保留；只有代码/名称冲突、非当前 A 股、路径错绑、证据伪造、业务明确相反或清晰无关时才会硬性剔除。CNFinancial 的空结果、报错或不可用会继续进入严格 Web fallback，而不会直接清空候选。

## 本地安装

```powershell
python -m pip install -r requirements.txt
```

运行全部测试：

```powershell
python -m pytest
```

如果当前环境没有安装 `pytest`，可以使用兼容的 unittest 命令：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -v
```

## 构建政策知识库

构建最小样例库：

```powershell
python scripts/ingest_sample.py --db data/processed/policychain.sqlite --reset
```

从本地政策目录构建完整库：

```powershell
python scripts/ingest_policy_dir.py --source-dir "D:\Code\人工智能政策文件" --manifest "D:\Code\人工智能政策文件\政策文件清单.csv" --db data/processed/policychain_full.sqlite --reset
```

数据库选择顺序：

1. `POLICYCHAIN_DB` 环境变量。
2. `data/processed/policychain_full.sqlite`。
3. `data/processed/policychain.sqlite`，必要时自动构建样例库。

## 本地 Web 使用

启动网页：

```powershell
python app.py
```

打开：

```text
http://127.0.0.1:8000
```

指定端口：

```powershell
$env:POLICYCHAIN_PORT='8010'
python app.py
```

指定完整政策库：

```powershell
$env:POLICYCHAIN_DB='data\processed\policychain_full.sqlite'
python app.py
```

## CLI 使用

使用样例库运行：

```powershell
python scripts/run_research.py --sample-db --query "粘贴政策正文或政策问题"
```

使用完整库运行：

```powershell
python scripts/run_research.py --full-db --query "粘贴政策正文或政策问题"
```

输出报告到文件：

```powershell
python scripts/run_research.py --full-db --query "粘贴政策正文或政策问题" --out artifacts/test-results/report.md
```

## DeepSeek 配置

本项目默认使用 DeepSeek。请使用环境变量或平台 Secrets 配置 API Key，不要把 API Key 写入代码。确定性流程仅作为显式 `--no-llm` 模式或 DeepSeek 不可用时的回退。

本地 PowerShell 示例：

```powershell
$env:POLICYCHAIN_LLM_PROVIDER='deepseek'
$env:DEEPSEEK_API_KEY='your-deepseek-api-key'
$env:DEEPSEEK_MODEL='deepseek-v4-flash'
$env:DEEPSEEK_THINKING='disabled'
```

可选配置：

```powershell
$env:DEEPSEEK_BASE_URL='https://api.deepseek.com'
$env:DEEPSEEK_TIMEOUT_SECONDS='60'
$env:DEEPSEEK_MAX_TOKENS='4096'
$env:DEEPSEEK_REASONING_EFFORT='high'
$env:DEEPSEEK_USE_SYSTEM_PROXY='1'
```

在 Hugging Face Spaces 中配置 DeepSeek：

1. 打开 Space 的 `Settings`。
2. 找到 `Variables and secrets`。
3. 点击 `New secret`。
4. Name 填写 `DEEPSEEK_API_KEY`。
5. Value 填写你的 DeepSeek API Key。
6. 保存后等待 Space 自动重启。

## MCP 配置

当前预留并支持两个 MCP 通道：

- Open-WebSearch：用于政策、行业外部证据，以及 Company Matcher 按路径发现具体公司和业务资料。
- CNFinancial：Impact Analyst 可使用行业和宏观资料；Company Matcher 只用具体公司名称或六位代码做身份、主营业务和增强资料核验，不再以宽泛行业关键词作为主要公司召回入口。

本地准备 MCP：

```powershell
python scripts/setup_mcp_servers.py
```

检查本地 MCP 配置：

```powershell
python scripts/mcp_doctor.py --mcp-config .mcp.local.json
```

运行 MCP smoke test：

```powershell
python scripts/mcp_smoke.py --mcp-config .mcp.local.json
```

CLI 启用 MCP：

```powershell
python scripts/run_research.py --full-db --mcp --query "粘贴政策正文或政策问题"
```

Hugging Face Spaces 云端默认关闭 MCP，原因是免费 CPU 上 stdio MCP 调用较慢。若要打开，在 Space 的 `Variables and secrets` 中添加 Variable：

```text
POLICYCHAIN_ENABLE_MCP_BY_DEFAULT=1
```

建议同时保留快速模式变量：

```text
POLICYCHAIN_MCP_FAST_MODE=1
POLICYCHAIN_MCP_MAX_POLICY_WEB_TOPICS=1
POLICYCHAIN_MCP_MAX_SELECTED_INDUSTRIES=2
POLICYCHAIN_MCP_MAX_SEARCH_TERMS=2
POLICYCHAIN_COMPANY_DISCOVERY_MODE=web_first
POLICYCHAIN_MCP_MAX_COMPANY_CANDIDATES=5
POLICYCHAIN_MAX_COMPANY_MATCHES_PER_IMPACT=3
POLICYCHAIN_MCP_COMPANY_ENRICH_TOOLS=get_company_profile
```

如果开启 MCP 后任务长时间停在行业影响或公司匹配阶段，请把：

```text
POLICYCHAIN_ENABLE_MCP_BY_DEFAULT=0
```

## Hugging Face Spaces 部署

仓库根目录 `README.md` 已包含 Hugging Face Spaces Docker front matter：

```yaml
sdk: docker
app_port: 10000
```

部署时建议设置 Variables：

```text
POLICYCHAIN_HOST=0.0.0.0
PORT=10000
POLICYCHAIN_DB=/app/data/processed/policychain_full.sqlite
POLICYCHAIN_MCP_CONFIG=/app/.mcp.local.json
POLICYCHAIN_LLM_PROVIDER=deepseek
POLICYCHAIN_MCP_TIMEOUT=90
POLICYCHAIN_ENABLE_MCP_BY_DEFAULT=0
POLICYCHAIN_MCP_FAST_MODE=1
```

设置 Secret：

```text
DEEPSEEK_API_KEY
```

注意：Secrets 用于 API Key 等敏感值；Variables 用于普通运行参数。不要把 `.env.local`、`.mcp.local.json`、API Key、Cookie 或 Token 提交到 Git。

## 项目结构

```text
policychain/
  agents/          三个 Agent 及 LLM 版本
  ingestion/       离线政策读取、编号、切分和入库
  schemas/         核心数据结构
  storage/         SQLite 存储
  tools/           本地 RAG 工具、MCP 工具和公司工具
scripts/           入库、运行、MCP 检查脚本
tests/             单元测试和工作流测试
data/sample/       最小样例政策数据
data/processed/    SQLite 数据库输出目录
```

## 三 Agent 分工

- Policy Analyst：识别政策标题、发布主体、发布日期、文号、政策目标、政策措施、约束对象、历史变化和政策力度。
- Impact Analyst：从政策措施推导实施主体、实施行为、产业链环节、行业经营变量和影响公司类型。
- Company Matcher：按行业影响路径使用 DeepSeek/Web 发现 A 股公司，以 CNFinancial 精确核验身份和主营资料，再融合证据、评价、固定复核和排序；保留业务证据、反面证据、风险、资料状态和置信度。

本地政策知识库只用于相似政策、历史政策或可比政策检索，不会替代用户输入政策本身。

## 当前限制

- 云端 MCP 默认关闭，避免 Hugging Face 免费 CPU 上任务超时。
- CNINFO 年报下载暂未作为默认流程启用。
- 当前公司部分是业务匹配研究清单，不做投资建议。
- 运行事件和摘要默认保存到 Git 忽略的 `artifacts/run-logs/<run_id>/`，并可按 job_id 或 run_id 下载；服务重启后内存中的 job 索引会重置，但已落盘的 run artifact 仍可按 run_id 读取。
