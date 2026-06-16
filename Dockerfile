FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    POLICYCHAIN_HOST=0.0.0.0 \
    POLICYCHAIN_MCP_CONFIG=/app/.mcp.local.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/setup_mcp_servers.py --config /app/.mcp.local.json

EXPOSE 10000

CMD ["python", "app.py"]
