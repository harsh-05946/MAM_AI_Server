# Queued router (`router_api.py`)

The router listens on **`PORT`** (default **9000**, see [`start_router_api.sh`](../start_router_api.sh)). It accepts the same paths as the main inference app and forwards each request to **`BACKEND_A`** or **`BACKEND_B`** (defaults `http://127.0.0.1:8001` and `:8002`).

## Round-robin (default)

Unless pinning is enabled, each queued request alternates **A → B → A → …** (`_next_backend`). Per-backend concurrency is limited by semaphores **`BACKEND_A_CAPACITY`** and **`BACKEND_B_CAPACITY`** (default `2` each in `start_router_api.sh`).

## Path pinning (optional)

When **`ROUTER_PIN_BACKEND`** is set to **`A`** or **`B`**, any request whose path **starts with** one of the configured prefixes is sent **only** to that backend. The round-robin counter is **not** advanced for pinned requests (RR applies only to unpinned traffic).

| Environment variable | Meaning |
|----------------------|--------|
| `ROUTER_PIN_BACKEND` | `A` or `B` (case-insensitive). Empty / unset = pinning disabled, pure RR. |
| `ROUTER_PIN_PATH_PREFIXES` | Comma-separated path prefixes. If unset while `ROUTER_PIN_BACKEND` is set, defaults to `/process/emotion,/process/embeddings` (covers `/process/emotion/batch` because it shares the `/process/emotion` prefix). |

### Why pin?

With **two identical backends**, server-side **micro-batching** (emotion, embeddings singles) builds queues **per process**. Strict RR splits concurrent singles across A and B, so each queue sees fewer arrivals. Pinning emotion and embedding routes to **one** backend concentrates load and can improve batch formation when both processes share one GPU or you want predictable batching behavior.

### Other env vars

| Variable | Default | Role |
|----------|---------|------|
| `BACKEND_A` | `http://127.0.0.1:8001` | Upstream base URL |
| `BACKEND_B` | `http://127.0.0.1:8002` | Upstream base URL |
| `ROUTER_QUEUE_MAX` | `200` | Max queued requests |
| `ROUTER_QUEUE_WAIT_TIMEOUT_SEC` | `300` | Client wait in queue |
| `ROUTER_UPSTREAM_TIMEOUT_SEC` | `300` | httpx timeout to upstream |
| `BACKEND_A_CAPACITY` | `2` | Max concurrent upstream requests to A |
| `BACKEND_B_CAPACITY` | `2` | Max concurrent upstream requests to B |

`GET /health` on the router includes `router_pin_backend` and `router_pin_path_prefixes` for verification.
