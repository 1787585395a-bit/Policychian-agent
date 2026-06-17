---
title: PolicyChain
sdk: docker
app_port: 10000
---

# PolicyChain

PolicyChain is an Agentic RAG project for policy research. The first-stage codebase focuses on offline policy ingestion: reading policy files, generating stable IDs, extracting metadata, chunking text, and validating the pipeline with tests.

## Setup

```powershell
python -m pip install -r requirements.txt
```

LangChain is optional. The built-in lightweight ReAct retrieval works without it. To install the optional LangChain wrappers, first make sure pip can access an index, then install:

```powershell
$env:PIP_NO_INDEX="0"
python -m pip install -r requirements-langchain.txt
```

## Tests

```powershell
python -m pytest
```

If `pytest` is not installed in the current environment, run the same unittest-compatible suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -v
```

## Sample Knowledge Base

Build or rebuild the sample SQLite policy database:

```powershell
python scripts/ingest_sample.py --db data/processed/policychain.sqlite --reset
```

Build or rebuild the full local policy database from `D:\Code\人工智能政策文件`:

```powershell
python scripts/ingest_policy_dir.py --source-dir "D:\Code\人工智能政策文件" --manifest "D:\Code\人工智能政策文件\政策文件清单.csv" --db data/processed/policychain_full.sqlite --reset
```

The research CLI and web app use this database selection order:

1. `POLICYCHAIN_DB` environment variable, when set.
2. `data/processed/policychain_full.sqlite`, when it exists.
3. `data/processed/policychain.sqlite`, rebuilding the one-policy sample when needed.

Smoke-check the policy tools after building the database:

```powershell
python -c "from policychain.storage import SQLitePolicyStore; from policychain.tools import search_policy; s=SQLitePolicyStore('data/processed/policychain.sqlite'); print(search_policy(s, '生成式人工智能', top_k=1)); s.close()"
```

Run the complete sample research workflow and print a Markdown report:

```powershell
python scripts/run_research.py --query "生成式人工智能服务提供者有哪些管理要求" --rebuild-sample-db
```

Run against the one-policy sample database explicitly:

```powershell
python scripts/run_research.py --sample-db --query "生成式人工智能服务提供者有哪些管理要求"
```

Run against the full local policy database explicitly:

```powershell
python scripts/run_research.py --full-db --query "生成式人工智能服务提供者有哪些管理要求"
```

Run the optional DeepSeek-backed workflow after configuring `.env.local` or environment variables:

```powershell
python scripts/run_research.py --full-db --query "生成式人工智能服务提供者有哪些管理要求" --llm
```

Write the Markdown report to a file:

```powershell
python scripts/run_research.py --out artifacts/test-results/sample_report.md
```

## DeepSeek LLM Provider

The project defaults to the offline `MockLLMClient`, so tests and local sample workflows do not require network access or an API key.

To create a DeepSeek client from environment variables:

```powershell
$env:POLICYCHAIN_LLM_PROVIDER='deepseek'
$env:DEEPSEEK_API_KEY='your-deepseek-api-key'
$env:DEEPSEEK_MODEL='deepseek-v4-flash'
$env:DEEPSEEK_THINKING='disabled'
```

Optional settings:

```powershell
$env:DEEPSEEK_BASE_URL='https://api.deepseek.com'
$env:DEEPSEEK_TIMEOUT_SECONDS='60'
$env:DEEPSEEK_MAX_TOKENS='4096'
$env:DEEPSEEK_REASONING_EFFORT='high'
$env:DEEPSEEK_USE_SYSTEM_PROXY='1'
```

`DeepSeekClient` uses the standard-library HTTP stack and calls `/chat/completions`. It ignores system proxy variables by default to avoid local proxy ports breaking API calls; set `DEEPSEEK_USE_SYSTEM_PROXY=1` only when you intentionally want Python to use the system proxy. It is available as a provider boundary only; the deterministic sample Agent workflow is still the default execution path.

Run the optional LLM Policy Analyst from Python code after configuring a provider:

```python
from policychain.agents import run_llm_policy_analyst
from policychain.state import PolicyResearchState

state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
run_llm_policy_analyst(state, store)
```

The default CLI, web app, and `run_policy_research_workflow` still use the deterministic Policy Analyst.

The optional LLM Impact Analyst can be run after `state.policy_analysis` and `state.policy_chunks` are populated:

```python
from policychain.agents import run_llm_impact_analyst

run_llm_impact_analyst(state)
```

The optional LLM Company Matcher can be run after `state.industry_impacts` is populated:

```python
from policychain.agents import run_llm_company_matcher

run_llm_company_matcher(state)
```

Run the optional LLM-backed workflow from Python code:

```python
from policychain.graph import run_llm_policy_research_workflow

state = run_llm_policy_research_workflow(
    "生成式人工智能服务提供者有哪些管理要求",
    store,
)
print(state.final_report)
```

## Structured LLM Output

LLM responses must pass through `policychain.structured_output` before they can be used as Agent outputs. The parser accepts raw JSON objects and fenced JSON blocks, then validates the result against supported schemas:

- `PolicyAnalysisOutput`
- `ImpactAnalysisOutput`
- `CompanyMatchOutput`

The validator rejects malformed JSON, missing required fields, invalid enum values, invalid confidence scores, and prohibited investment-advice terms.

## MCP Evidence Interfaces

The default runtime uses `UnavailableMCPInvoker`, so local tests remain offline and deterministic. Real stdio MCP calls are enabled only when CLI commands pass `--mcp`; setup generates `.mcp.local.json` and prepares the local runtime dependencies.

Reserved MCP servers and tools:

- Open-WebSearch `web-search`: `search`, `fetchWebContent`
- CNFinancial `cn-financial`: selected industry, macro, company, financial, announcement, and news tools

Agent responsibilities:

- Policy Analyst keeps local `search_policy`, `get_policy_metadata`, and `read_policy_content` as primary evidence, with Web Search only as supplemental policy evidence.
- Impact Analyst can receive CNFinancial industry/macro/news evidence plus Web Search industry evidence, and must express `policy measure -> implementation action -> chain segment -> business variables -> affected company types`.
- Company Matcher uses CNFinancial for A-share candidate screening and public company evidence, with Web Search only as supplemental official-announcement/company-site evidence.

Prepare local stdio MCP servers and generate `.mcp.local.json`:

```powershell
python scripts/setup_mcp_servers.py
```

The setup script clones or updates CNFinancial under `external/mcp/`, installs its runtime dependencies, and generates a local config for Open-WebSearch and CNFinancial.

Check the local MCP config without making live network calls:

```powershell
python scripts/mcp_doctor.py --mcp-config .mcp.local.json
```

Smoke-check the configured MCP servers:

```powershell
python scripts/mcp_smoke.py --mcp-config .mcp.local.json
```

Run the deterministic workflow with real MCP tools:

```powershell
python scripts/run_research.py --full-db --mcp --query "生成式人工智能服务提供者有哪些管理要求" --out artifacts/test-results/mcp_live_report.md
```

Real MCP workflow runs use a small in-process cache for repeated identical calls. Add `--no-mcp-cache` when debugging raw MCP server behavior.

Run the DeepSeek-backed workflow with real MCP tools:

```powershell
python scripts/run_research.py --full-db --llm --mcp --query "生成式人工智能服务提供者有哪些管理要求" --out artifacts/test-results/mcp_live_deepseek_report.md
```

Local MCP files and generated artifacts are ignored by git: `.mcp.local.json`, `external/mcp/`, and `artifacts/`.

Run the local web app:

```powershell
python app.py
```

Open `http://127.0.0.1:8000`.

To force the web app to use the full local database:

```powershell
$env:POLICYCHAIN_DB="data\processed\policychain_full.sqlite"
python app.py
```

## Cloud Deployment on Hugging Face Spaces

This repository is configured for Hugging Face Spaces Docker deployment through the README front matter:

```yaml
sdk: docker
app_port: 10000
```

Create a Hugging Face Space with SDK `Docker` and hardware `CPU Basic`, then push this repository to the Space git remote or connect it from GitHub.

Set these Space variables:

- `POLICYCHAIN_HOST=0.0.0.0`
- `PORT=10000`
- `POLICYCHAIN_DB=/app/data/processed/policychain_full.sqlite`
- `POLICYCHAIN_MCP_CONFIG=/app/.mcp.local.json`
- `POLICYCHAIN_LLM_PROVIDER=deepseek`
- `POLICYCHAIN_MCP_TIMEOUT=90`

Set this Space secret:

- `DEEPSEEK_API_KEY`

The Docker build installs Node/npm/git, Python dependencies, and prepares the stdio MCP config at `/app/.mcp.local.json`. Open-WebSearch and CNFinancial remain MCP tools; they are not replaced with direct Python functions.

After deployment, check:

```text
https://<space-owner>-policychain-agent.hf.space/healthz
```

Then submit a policy URL and confirm the runtime logs show MCP initialization instead of a missing MCP config fallback.

## Cloud Deployment on Render

This repository includes a Dockerfile and `render.yaml` for deploying PolicyChain as a public Render Web Service. The first deployment uses Render's default `*.onrender.com` URL and does not configure a custom domain.

Before deploying, make sure the full SQLite database exists:

```powershell
python scripts/ingest_policy_dir.py --source-dir "D:\Code\人工智能政策文件" --manifest "D:\Code\人工智能政策文件\政策文件清单.csv" --db data/processed/policychain_full.sqlite --reset
```

Local Docker smoke check:

```powershell
docker build -t policychain-web .
docker run --rm -p 8010:10000 -e PORT=10000 -e POLICYCHAIN_HOST=0.0.0.0 policychain-web
```

Open:

```text
http://127.0.0.1:8010/
http://127.0.0.1:8010/healthz
```

Render setup:

1. Push the repository to GitHub.
2. In Render, create a new Blueprint or Docker Web Service from this repository.
3. Use the service name `policychain-agent` if available; otherwise use `policychain-research` or `policychain-rag`.
4. Set `DEEPSEEK_API_KEY` in Render environment variables. Do not commit it to git.
5. Deploy and open the generated Render default URL.

The Render service uses:

- `POLICYCHAIN_HOST=0.0.0.0`
- `POLICYCHAIN_DB=/app/data/processed/policychain_full.sqlite`
- `POLICYCHAIN_MCP_CONFIG=/app/.mcp.local.json`
- `POLICYCHAIN_LLM_PROVIDER=deepseek`

The site is public by default. Anyone with the Render URL can submit analyses and trigger DeepSeek/MCP usage.

## Current Stage

Implemented scope:

- Core dataclass schemas and shared research state.
- Offline ingestion skeleton for PDF, Markdown, and text files.
- Stable file hash, policy ID, and chunk ID generation.
- Sample policy manifest and one real PDF sample under `data/sample/raw/`.
- SQLite storage for ingested policies and chunks.
- Repeatable sample database builder under `scripts/ingest_sample.py`.
- Full local policy database builder under `scripts/ingest_policy_dir.py`.
- Basic policy tools: `search_policy`, `get_policy_metadata`, and `read_policy_content`.
- Deterministic Policy Analyst output over retrieved policy chunks.
- Deterministic Impact Analyst output over Policy Analyst results.
- Deterministic Company Matcher over local mock company data.
- Markdown report writer and end-to-end workflow runner.
- CLI runner for the sample research workflow.
- No-dependency local web UI for the sample workflow.
- DeepSeek-compatible LLM provider boundary with offline unit tests.
- Structured LLM JSON parsing and schema validation boundary.
- Optional LLM Policy Analyst runner over retrieved policy evidence.
- Optional LLM Impact Analyst runner over Policy Analyst output.
- Optional LLM Company Matcher runner over retrieved company candidates.
- Optional LLM-backed workflow orchestrator.
- Default database selection that prefers the full local policy database when it exists.
- Unit tests for loading, IDs, metadata extraction, chunking, ingestion, storage, tools, Agents, LLM boundaries, and workflows.

Not implemented yet:

- Vector store, BM25, and hybrid retrieval.
- Real company data search.
- Browser-driven UI checks in CI.
