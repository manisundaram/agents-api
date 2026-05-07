# agents-api

`agents-api` is a production-ready FastAPI microservice for building AI agent workflows, built on top of the shared [`ai-service-kit`](../ai-service-kit/README.md) library. It provides ReAct-style single agents, multi-agent workflows, cost-optimized model routing, semantic caching, and full operational observability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT / UI                              │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      agents-api  :8000                          │
│                                                                 │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │  /agent/    │   │  Model Router    │   │ Semantic Cache  │  │
│  │  query      │──▶│  cheap / expensive│──▶│ (vector similarity)│
│  │  multi      │   │  fallback logic  │   └─────────────────┘  │
│  └──────┬──────┘   └──────────────────┘                        │
│         │                                                       │
│  ┌──────▼──────────────────────────────────────┐               │
│  │             Agent Execution Layer            │               │
│  │                                             │               │
│  │  ┌──────────────┐    ┌────────────────────┐ │               │
│  │  │  Single Agent│    │   Multi-Agent      │ │               │
│  │  │  (ReAct loop)│    │  Planner           │ │               │
│  │  │              │    │    → Worker(s)     │ │               │
│  │  │              │    │    → Reviewer      │ │               │
│  │  └──────┬───────┘    └────────┬───────────┘ │               │
│  └─────────┼────────────────────┼─────────────┘               │
│            │   Tools            │                              │
│  ┌─────────▼────────────────────▼─────────────┐               │
│  │                  Tool Layer                 │               │
│  │  calculator │ file_reader │ search │ rag    │               │
│  └──────────────────┬─────────────┬────────────┘               │
│                     │             │                             │
│  ┌──────────────────▼──┐  ┌───────▼──────────┐                │
│  │   LLM Providers     │  │ External Services │                │
│  │  app/providers/     │  │                  │                │
│  │  ┌───────────────┐  │  │  semantic-search  │                │
│  │  │ OpenAIProvider│  │  │  :8001            │                │
│  │  │ (via factory) │  │  │                  │                │
│  │  └───────────────┘  │  │  rag-api          │                │
│  └─────────────────────┘  │  :8002            │                │
│                            └───────────────────┘               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  ai-service-kit  (shared library)        │   │
│  │   Health │ Logging │ Diagnostics │ Metrics │ Embeddings  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Service Map

| Service               | Port | Role                                               |
| --------------------- | ---- | -------------------------------------------------- |
| `agents-api`          | 8000 | This service — agent orchestration and LLM routing |
| `semantic-search-api` | 8001 | Optional — vector search over indexed documents    |
| `rag-api`             | 8002 | Optional — retrieval-augmented generation pipeline |

> `semantic-search-api` and `rag-api` are separate microservices in this portfolio. `agents-api` works standalone without them; health checks will report `degraded` on `external_services` if they are not running.

---

## What this service provides

- **Single Agent**: ReAct-style reasoning loops with tool use and step traces
- **Multi-Agent Orchestration**: Planner → Worker(s) → Reviewer pipeline with structured results
- **LLM Provider Layer**: Custom `app/providers/` module with `OpenAIProvider` and an extensible factory (supports OpenAI, Gemini, Claude)
- **Model Routing**: Cost-optimized routing — cheap model for simple tasks, expensive model for complex ones, with automatic fallback
- **Semantic Caching**: Vector similarity cache to skip redundant LLM calls and reduce cost
- **Guardrails**: Input/output validation, PII filtering, and safety wrappers
- **Memory Management**: In-loop and persistent agent memory
- **Full Observability**: Health checks, diagnostics, metrics, and debug endpoints

## How it uses ai-service-kit

`agents-api` depends on the shared [`ai-service-kit`](../ai-service-kit) library (editable install from `../ai-service-kit`). It uses:

- **Health Monitoring**: `BaseHealthCheck`, `ServiceContext`, `HealthStatus` — wired in `app/bootstrap.py`
- **Enhanced Logging**: Structured JSON logs with request correlation and optional cloud provider integration
- **Diagnostics**: Deep system analysis with performance benchmarks via `/diagnostics`
- **Embeddings**: `ProviderFactory` for vector embeddings used by `semantic_cache.py`
- **Utilities**: `mask_secret`, `LoggingMiddleware`, configuration helpers

> **Note**: `ai-service-kit` does **not** provide a general LLM inference `create_provider`. That is handled by `app/providers/` in this project.

## Project structure

```text
agents-api/
├── .env                      # Local environment config (not committed)
├── .env.example              # Environment configuration template
├── pyproject.toml            # Python project configuration
├── requirements.txt          # Python dependencies
├── app/
│   ├── main.py               # FastAPI app factory — all endpoints registered here
│   ├── config.py             # Settings (pydantic-settings, reads from .env)
│   ├── bootstrap.py          # ServiceContext builder with health checks
│   ├── models.py             # Pydantic request/response models
│   ├── agent.py              # Single-agent ReAct reasoning loop
│   ├── multi_agent.py        # Multi-agent planner → worker → reviewer
│   ├── router.py             # Cost-optimized model routing with fallback
│   ├── semantic_cache.py     # Vector similarity response cache
│   ├── guardrails.py         # Input/output validation and safety filters
│   ├── memory.py             # Agent memory management
│   ├── providers/            # Custom LLM provider layer
│   │   ├── base.py           # BaseLLMProvider + LLMResponse dataclass
│   │   ├── factory.py        # create_provider(provider_type, config) factory
│   │   └── openai_provider.py# OpenAI async chat completions implementation
│   └── tools/                # Agent tools
│       ├── calculator.py     # Safe math expression evaluator
│       ├── file_reader.py    # Local file reading
│       ├── search_tool.py    # HTTP client for semantic-search-api
│       └── rag_tool.py       # HTTP client for rag-api
└── tests/
    └── test_app.py           # Integration tests for all endpoints (7 tests)
```

## API Endpoints

### Agent Endpoints

| Method | Path           | Description                                      |
| ------ | -------------- | ------------------------------------------------ |
| `POST` | `/agent/query` | Single-agent ReAct reasoning loop                |
| `POST` | `/agent/multi` | Multi-agent planner → worker → reviewer workflow |
| `GET`  | `/agent/tools` | List all available tools for agent execution     |
| `GET`  | `/agent/trace` | Debug — inspect the last agent run trace         |

### Operational Endpoints

| Method | Path                   | Description                                           |
| ------ | ---------------------- | ----------------------------------------------------- |
| `GET`  | `/ping`                | Lightweight liveness check (for load balancers)       |
| `GET`  | `/health`              | Comprehensive health report with per-component status |
| `GET`  | `/diagnostics`         | Deep system analysis with API tests and benchmarks    |
| `GET`  | `/metrics`             | Prometheus-style metrics for monitoring dashboards    |
| `GET`  | `/debug/config`        | Configuration introspection with masked secrets       |
| `POST` | `/debug/cache/clear`   | Clear the semantic cache                              |
| `GET`  | `/debug/router/stats`  | Model router usage statistics                         |
| `GET`  | `/debug/router/health` | Model router health status                            |

---

## Dependencies

### Core

| Package                          | Purpose                                             |
| -------------------------------- | --------------------------------------------------- |
| `fastapi`                        | Web framework and request routing                   |
| `uvicorn[standard]`              | ASGI server                                         |
| `pydantic` + `pydantic-settings` | Data validation and environment config              |
| `python-dotenv`                  | `.env` file loading                                 |
| `httpx`                          | Async HTTP client (tool calls to external services) |
| `openai`                         | OpenAI async SDK used by `OpenAIProvider`           |
| `numpy`                          | Vector math for semantic cache similarity scoring   |

### Shared Library

| Package          | Source                                                                                        |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `ai-service-kit` | Editable install from `../ai-service-kit` — provides health, logging, diagnostics, embeddings |

### Dev / Test

| Package  | Purpose                            |
| -------- | ---------------------------------- |
| `pytest` | Test runner                        |
| `httpx`  | Also used by `TestClient` in tests |

### Optional Cloud Logging (production)

```bash
pip install boto3>=1.26.0               # AWS CloudWatch
pip install applicationinsights>=0.11.0 # Azure Monitor
pip install google-cloud-logging>=3.0.0 # Google Cloud
pip install datadog>=0.44.0             # Datadog
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- The [`ai-service-kit`](../ai-service-kit) repository cloned at `../ai-service-kit` (sibling directory)
- An OpenAI API key (or Gemini / Anthropic)

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `ai-service-kit` as an editable dependency from `../ai-service-kit` and all core packages.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-...
```

See the full [Configuration](#configuration) section below for all options.

### 4. Start the service

```bash
# Development (auto-reload on file changes)
uvicorn app.main:app --reload --port 8000

# Using the .venv Python directly (recommended on Windows)
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

### 5. Verify it's running

```bash
curl http://localhost:8000/ping
# {"status":"ok","service_name":"agents-api","service_version":"0.1.0","timestamp":"..."}

curl http://localhost:8000/agent/tools
# {"tools": [...]}
```

### 6. Run tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
# 7 passed
```

---

## Quick Example

**Single agent query:**

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 42 * 17?", "tools": ["calculator"], "max_steps": 3}'
```

**Multi-agent workflow:**

```bash
curl -X POST http://localhost:8000/agent/multi \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain quantum entanglement simply", "max_steps": 5}'
```

---

## Configuration

All configuration is via environment variables loaded from `.env`. Copy `.env.example` to get started.

### Provider

```env
PROVIDER_TYPE=openai                  # openai | gemini | claude

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash

ANTHROPIC_API_KEY=...
CLAUDE_MODEL=claude-3-5-haiku-latest
```

### Agent

```env
AGENT_MAX_STEPS=10                    # Max ReAct loop iterations
AGENT_TEMPERATURE=0.1                 # LLM sampling temperature
AGENT_TIMEOUT_SECONDS=300             # Execution timeout
```

### Model Routing

```env
CHEAP_MODEL_PROVIDER=openai
CHEAP_MODEL=gpt-4o-mini              # Used for simple/fast tasks

EXPENSIVE_MODEL_PROVIDER=openai
EXPENSIVE_MODEL=gpt-4o               # Used for complex reasoning
```

### Semantic Cache

```env
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.85        # Cosine similarity cutoff (0–1)
SEMANTIC_CACHE_MAX_ENTRIES=1000
```

### External Services (optional)

```env
SEMANTIC_SEARCH_API_URL=http://localhost:8001
RAG_API_URL=http://localhost:8002
```

If these services are not running, `agents-api` still works — only `/health` will report `external_services` as degraded.

### CORS

```env
ENABLE_CORS=true
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
```

---

## Production Deployment

The service includes enterprise-grade features for production:

- **Structured Logging**: JSON logs via `ai-service-kit` with request correlation IDs
- **Cloud Logging**: Optional AWS CloudWatch, Datadog, Azure Monitor, GCP integration (set `CLOUD_LOGGING_PROVIDERS`)
- **Health Endpoints**: `/health` and `/diagnostics` for readiness/liveness probes
- **Metrics**: `/metrics` for Prometheus scraping or dashboard integration
- **Secret Masking**: All API keys are masked in logs and debug output
- **Semantic Cache**: Reduces LLM call volume and latency for repeated queries
- **Model Routing**: Automatic fallback from expensive to cheap model on failure

See [ai-service-kit documentation](../ai-service-kit/README.md) for cloud logging setup and deployment patterns.
