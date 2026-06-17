FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    NPM_CONFIG_CACHE=/tmp/npm-cache \
    PIP_CACHE_DIR=/tmp/pip-cache \
    PORT=10000 \
    POLICYCHAIN_HOST=0.0.0.0 \
    POLICYCHAIN_MCP_CONFIG=/app/.mcp.local.json \
    POLICYCHAIN_MCP_TIMEOUT=90 \
    POLICYCHAIN_MCP_FAST_MODE=1 \
    POLICYCHAIN_MCP_MAX_POLICY_WEB_TOPICS=1 \
    POLICYCHAIN_MCP_MAX_SELECTED_INDUSTRIES=2 \
    POLICYCHAIN_MCP_MAX_SEARCH_TERMS=2 \
    POLICYCHAIN_MCP_MAX_COMPANY_CANDIDATES=1 \
    POLICYCHAIN_MCP_COMPANY_ENRICH_TOOLS=get_company_profile

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user \
    && mkdir -p /tmp/npm-cache /tmp/pip-cache \
    && chown -R user:user /tmp/npm-cache /tmp/pip-cache

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/setup_mcp_servers.py --config /app/.mcp.local.json \
    && chown -R user:user /app /tmp/npm-cache /tmp/pip-cache

USER user

EXPOSE 10000

CMD ["python", "app.py"]
