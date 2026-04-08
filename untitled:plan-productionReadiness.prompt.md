## Plan: Production Hardening for AI Server

TL;DR: Evolve the current FastAPI demo into a production-ready AI service by making model loading predictable, isolating costly workloads, adding request control and observability, and preparing deployment infrastructure for GPU-backed inference.

**Steps**
1. Audit current app behavior and workload.
   - Review `app/main.py` and `app/models.py` to identify heavy endpoints and shared resources.
   - Document which endpoints require GPU vs CPU and which are I/O-bound.

2. Add centralized configuration.
   - Create `app/config.py` for environment-driven values: `HOST`, `PORT`, `LOG_LEVEL`, `MAX_WORKERS`, `QUEUE_SIZE`, `RATE_LIMIT`, `MODEL_DEVICE`, `CUDA_VISIBLE_DEVICES`, `CACHE_TTL`, `SENTRY_DSN`.
   - Replace hard-coded values in `main.py`, `models.py`, and `start_api.sh` with config-driven values.

3. Separate model loading and request handling.
   - Keep `app/models.py` responsible for lazy, thread-safe model initialization and clean shutdown.
   - Add readiness/health endpoints in `app/main.py`: `/health/live`, `/health/ready`, `/metrics`.
   - Ensure startup does not block if optional models fail to load; expose degraded status.

4. Control expensive inference concurrency.
   - Add a bounded request queue or worker pool for GPU-heavy endpoints such as `/process/ram-tags`, `/process/scene`, `/process/transcription`.
   - Use `ThreadPoolExecutor` or an async queue with explicit `max_workers` separate from `uvicorn` workers.
   - Reject or queue excess requests gracefully with 429 responses.

5. Avoid shared GPU state across multiple processes.
   - Keep GPU-backed service single-process for RAM and Whisper models; do not use `uvicorn --workers` with GPU models unless using multi-process GPU-aware orchestration.
   - For CPU-only endpoints, consider a separate service or worker pool that can scale horizontally.

6. Add request validation and rate limiting.
   - Validate uploaded files, image MIME types, file sizes, and payload shapes before inference.
   - Use an API gateway or application-level rate limiter for per-client throttling.

7. Add caching and duplicate request handling.
   - Implement result caching keyed by image/audio content hash and endpoint type.
   - Store cache in Redis or local filesystem cache with TTL to reduce repeated expensive inference.

8. Improve observability and diagnostics.
   - Add structured logging, request IDs, endpoint latency, and error classification in `app/main.py`.
   - Expose Prometheus-compatible metrics for request count, latency, inference queue depth, model load success/failure.
   - Add health and readiness probes for container orchestration.

9. Harden deployment and runtime.
   - Dockerize the app with a `Dockerfile` and GPU runtime support if needed.
   - Use a process supervisor or Kubernetes with a single GPU-backed pod + autoscaling for CPU services.
   - Keep `start_api.sh` for local startup but use container startup commands in production.

10. Add testing and load verification.
   - Add unit/integration tests for endpoints and model loader health.
   - Create load tests for concurrency and threshold behavior using `send_audio.py` or a new script.
   - Verify memory usage and request latency under expected traffic.

**Relevant files**
- `app/main.py` — endpoint lifecycle, task queueing, request validation, logs, health checks.
- `app/models.py` — model loading strategy, device assignment, safe shutdown.
- `app/requirements.txt` — add production dependencies such as `prometheus-client`, `python-dotenv`, `gunicorn` or `uvicorn[standard]`, optional `redis`.
- `app/start_api.sh` — update startup for production-safe process settings.
- `app/config.py` (new) — centralized environment configuration.
- `Dockerfile` (new) — container build and runtime.

**Verification**
1. Start the app and validate `/health/live` and `/health/ready` return success.
2. Send concurrent requests to the expensive endpoints and verify queue limits, 429 handling, and stable latency.
3. Confirm GPU-backed models load once and do not duplicate across multiple workers in production.
4. Verify metrics visibility and that logs include request IDs and inference durations.

**Decisions**
- Single-process GPU inference is safer for RAM and Whisper; separate CPU/IO services if horizontal scaling is needed.
- Use explicit queueing rather than simply increasing `uvicorn` workers for heavy models.
- Add caching as a low-cost way to improve high-traffic behavior for repeated requests.

**Further Considerations**
1. Decide whether to split the app into multiple microservices or keep one monolith with internal task queues.
2. Choose between local cache vs Redis for production state.
3. Decide if upload size limits and content scanning are required for security.
