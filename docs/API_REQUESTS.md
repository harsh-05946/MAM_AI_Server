# API request shapes: single vs batch

Call the same paths on a **backend** (e.g. `http://127.0.0.1:8001`) or through the **router** (e.g. `http://127.0.0.1:9000`); the router forwards method, path, query, and body unchanged.

Quick checks: run [`tools/smoke_batch_endpoints.py`](../tools/smoke_batch_endpoints.py) against a running server.

Unless noted, responses are JSON.

---

## Legend

| Term | Meaning |
|------|--------|
| **Single (HTTP)** | One logical item per request body (one file or one JSON payload shape). |
| **Batch (HTTP)** | One request carries **multiple** items the server processes in one forward (where supported). |
| **Micro-batch (server)** | Multiple **concurrent** single-item HTTP requests are merged inside the server for a short window (`BATCH_MAX_*`, `BATCH_WAIT_MS_*`, `BATCHING_ENABLED`). No extra client fields required. |

---

## `POST /process/face`

| Mode | Content-Type | Body |
|------|----------------|------|
| Single (HTTP) | `multipart/form-data` | One part: form field **`file`** (image: jpeg/png/webp, etc.). |
| Batch (HTTP) | — | **Not implemented.** One image per request. |
| Micro-batch (server) | — | If `BATCHING_ENABLED=true`, concurrent face requests may be grouped (default `BATCH_MAX_FACE=4`, `BATCH_WAIT_MS_FACE=8`). Each image still runs `model.get()` sequentially inside the batch worker. |

**Example (curl, single):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/face" \
  -F "file=@/path/to/image.jpg"
```

**Response:** JSON array of `{ "bbox": [...], "embedding": [...] }` (one entry per detected face in that image).

---

## `POST /process/emotion`

| Mode | Content-Type | Body |
|------|----------------|------|
| Single (HTTP) | `multipart/form-data` | One part: form field **`file`** (one face crop or full frame as RGB image). |
| Batch (HTTP) | — | Use **`POST /process/emotion/batch`** (below). |
| Micro-batch (server) | — | Concurrent single-file emotion calls may merge (default `BATCH_MAX_EMOTION=16`, `BATCH_WAIT_MS_EMOTION=8`). |

**Example (curl, single):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/emotion" \
  -F "file=@/path/to/face_crop.png"
```

**Response:** `{ "emotion": "<label>", "confidence": <float> }`

---

## `POST /process/emotion/batch`

| Mode | Content-Type | Body |
|------|----------------|------|
| Batch (HTTP) | `multipart/form-data` | Multiple parts with the **same** field name **`files`** (one image per part). Order of parts = order of results. Max count: **`EMOTION_BATCH_MAX`** (default `32`). |

**Example (curl, two crops):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/emotion/batch" \
  -F "files=@/path/crop1.png" \
  -F "files=@/path/crop2.png"
```

**Response:** JSON **array** of `{ "emotion", "confidence", "filename"? }` objects, same length and order as uploaded files.

---

## `POST /process/scene`

| Mode | Content-Type | Body |
|------|----------------|------|
| Single (HTTP) | `multipart/form-data` | Form field **`file`**. |
| Batch (HTTP) | — | **Not implemented** (one image per request). |
| Micro-batch (server) | — | `BATCH_MAX_SCENE`, `BATCH_WAIT_MS_SCENE` (defaults 8 / 10 ms). |

**Example (curl):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/scene" \
  -F "file=@/path/to/image.jpg"
```

**Response:** `{ "scene": "<caption text>" }`

---

## `POST /process/object-detection` (RAM++)

| Mode | Content-Type | Body |
|------|----------------|------|
| Single (HTTP) | `multipart/form-data` | Form field **`file`**. |
| Batch (HTTP) | — | **Not implemented** (one image per request). |
| Micro-batch (server) | — | `BATCH_MAX_RAM_PLUS`, `BATCH_WAIT_MS_RAM_PLUS` (defaults 8 / 10 ms). |

**Example (curl):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/object-detection" \
  -F "file=@/path/to/image.jpg"
```

**Response:** `{ "tags_en": ..., "tags_cn": ... }` (shape depends on RAM++ output for that image).

---

## `POST /process/embeddings`

| Mode | Content-Type | Body |
|------|----------------|------|
| Single (HTTP) | `application/json` | `{ "texts": [ "one string" ] }` — may be **micro-batched** with other concurrent single-text requests. |
| Batch (HTTP) | `application/json` | `{ "texts": [ "s1", "s2", "s3", ... ] }` — **one** `SentenceTransformer.encode` over the full list (order preserved); **not** merged with other HTTP requests. |
| Micro-batch (server) | — | When `len(texts)==1` and `BATCHING_ENABLED=true`, concurrent requests each with one string can merge (`BATCH_MAX_EMBED`, `BATCH_WAIT_MS_EMBED`). For many strings you already hold, prefer one JSON with multiple `texts`. |

Optional: **`EMBED_ENCODE_BATCH_SIZE`** — if set to a positive integer, passed as `batch_size` to `encode()` for large lists.

**Example (curl, single string):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"texts":["hello world"]}'
```

**Example (curl, batch in one request):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"texts":["first text","second text","third text"]}'
```

**Response:** `{ "embeddings": [ [...], [...], ... ], "count": <int> }` — each inner list is one embedding vector (same order as `texts`).

---

## `POST /process/translation/sarvam`

| Mode | Content-Type | Body |
|------|----------------|------|
| Single (HTTP) | `application/json` | `{ "text": "<source text>", "target_lang": "Hindi" }` — `target_lang` may be a key from the server map (e.g. `hindi`) or a display name; see `SUPPORTED_LANGUAGES` in `main.py`. |
| Batch (HTTP) | — | **Not implemented** (one `text` per request). |

**Example (curl):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/translation/sarvam" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","target_lang":"hindi"}'
```

**Response:** `{ "translated_text": "...", "target_lang": "...", "engine": "sarvam" }`

---

## `POST /process/caption/qwen`

| Mode | Content-Type | Body |
|------|----------------|------|
| Single (HTTP) | `multipart/form-data` | **`file`**: image. **`prompt`**: form field (string); optional, defaults to built-in OCR-style prompt in code. |

**Example (curl):**

```bash
curl -sS -X POST "http://127.0.0.1:8001/process/caption/qwen" \
  -F "file=@/path/to/image.jpg" \
  -F "prompt=Extract all text present in the image."
```

**Response:** `{ "caption": "...", "model": "...", "prompt_used": "...", "duration_sec": ... }`

---

## `GET /health`

No body. Returns service status, loaded model keys, and optional CUDA memory stats.

---

## Environment knobs (micro-batching)

When `BATCHING_ENABLED` is true (default), these affect **server-side** merging of **separate** HTTP requests:

| Variable | Default | Models |
|----------|---------|--------|
| `BATCH_MAX_EMOTION` | 16 | emotion |
| `BATCH_WAIT_MS_EMOTION` | 8 | emotion |
| `BATCH_MAX_SCENE` | 8 | scene |
| `BATCH_WAIT_MS_SCENE` | 10 | scene |
| `BATCH_MAX_RAM_PLUS` | 8 | object-detection |
| `BATCH_WAIT_MS_RAM_PLUS` | 10 | object-detection |
| `BATCH_MAX_FACE` | 4 | face |
| `BATCH_WAIT_MS_FACE` | 8 | face |
| `BATCH_MAX_EMBED` | 32 | embeddings (single-text requests only) |
| `BATCH_WAIT_MS_EMBED` | 8 | embeddings |
| `EMOTION_BATCH_MAX` | 32 | max images per `POST /process/emotion/batch` |

Set `BATCHING_ENABLED=false` to disable micro-batching (each request runs immediately alone).

---

## Router (`router_api.py`)

- Same paths and bodies as above; target host is the router (e.g. port **9000**).
- Default: **round-robin** between `BACKEND_A` and `BACKEND_B` (can thin out per-process micro-batches for concurrent singles).
- Optional **path pinning**: set `ROUTER_PIN_BACKEND` and optionally `ROUTER_PIN_PATH_PREFIXES` — see [router.md](router.md).

Router `GET /health` returns `router_pin_backend` and `router_pin_path_prefixes` when pinning is configured.
