import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

logger = logging.getLogger("router")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


BACKEND_A = os.getenv("BACKEND_A", "http://127.0.0.1:8001")
BACKEND_B = os.getenv("BACKEND_B", "http://127.0.0.1:8002")

QUEUE_MAX = max(1, _env_int("ROUTER_QUEUE_MAX", 200))
QUEUE_WAIT_TIMEOUT_SEC = max(1, _env_int("ROUTER_QUEUE_WAIT_TIMEOUT_SEC", 300))
UPSTREAM_TIMEOUT_SEC = max(5, _env_int("ROUTER_UPSTREAM_TIMEOUT_SEC", 300))

BACKEND_A_CAPACITY = max(1, _env_int("BACKEND_A_CAPACITY", 2))
BACKEND_B_CAPACITY = max(1, _env_int("BACKEND_B_CAPACITY", 2))

# Optional: pin selected path prefixes to one backend (no RR for those paths).
# When ROUTER_PIN_BACKEND is A or B, defaults pin /process/emotion and /process/embeddings
# (override list with ROUTER_PIN_PATH_PREFIXES).
_ROUTER_PIN_RAW = os.getenv("ROUTER_PIN_BACKEND", "").strip().upper()
ROUTER_PIN_BACKEND = _ROUTER_PIN_RAW if _ROUTER_PIN_RAW in ("A", "B") else ""
if ROUTER_PIN_BACKEND:
    _pfx_default = "/process/emotion,/process/embeddings"
    _pfx_raw = os.getenv("ROUTER_PIN_PATH_PREFIXES", _pfx_default).strip()
    ROUTER_PIN_PATH_PREFIXES: tuple[str, ...] = tuple(
        p.strip() for p in _pfx_raw.split(",") if p.strip()
    )
else:
    ROUTER_PIN_PATH_PREFIXES = ()


@dataclass
class _QueuedRequest:
    request: Request
    body: bytes
    enqueued_at: float
    future: asyncio.Future


class _RouterState:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue(maxsize=QUEUE_MAX)
        self.a_sem = asyncio.Semaphore(BACKEND_A_CAPACITY)
        self.b_sem = asyncio.Semaphore(BACKEND_B_CAPACITY)
        self._rr_counter = 0
        self.inflight_a = 0
        self.inflight_b = 0
        self.started_at = time.time()
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT_SEC, connect=5.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="router-scheduler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def enqueue(self, request: Request, body: bytes) -> Response:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        item = _QueuedRequest(request=request, body=body, enqueued_at=time.time(), future=fut)

        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail="Router queue full")

        try:
            return await asyncio.wait_for(fut, timeout=QUEUE_WAIT_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.warning(
                "Queue wait timeout (%ss): path=%s queue_depth=%s inflight_a=%s inflight_b=%s",
                QUEUE_WAIT_TIMEOUT_SEC,
                request.url.path,
                self.queue.qsize(),
                self.inflight_a,
                self.inflight_b,
            )
            raise HTTPException(status_code=504, detail="Timed out waiting in queue")

    def _pinned_backend(self, path: str) -> Optional[tuple[str, asyncio.Semaphore]]:
        if not ROUTER_PIN_BACKEND or not ROUTER_PIN_PATH_PREFIXES:
            return None
        for prefix in ROUTER_PIN_PATH_PREFIXES:
            if path.startswith(prefix):
                if ROUTER_PIN_BACKEND == "A":
                    return BACKEND_A, self.a_sem
                return BACKEND_B, self.b_sem
        return None

    def _next_backend(self) -> tuple[str, asyncio.Semaphore]:
        # Strict round-robin by request order; no fallback.
        prefer_a = (self._rr_counter % 2 == 0)
        self._rr_counter += 1
        if prefer_a:
            return BACKEND_A, self.a_sem
        return BACKEND_B, self.b_sem

    async def _forward(self, backend_base: str, item: _QueuedRequest) -> Response:
        assert self._client is not None
        upstream_url = f"{backend_base}{item.request.url.path}"
        if item.request.url.query:
            upstream_url += f"?{item.request.url.query}"

        headers = dict(item.request.headers)
        headers.pop("host", None)

        resp = await self._client.request(
            method=item.request.method,
            url=upstream_url,
            content=item.body,
            headers=headers,
        )

        # Pass through status code and content-type; strip hop-by-hop headers.
        out_headers = {}
        content_type = resp.headers.get("content-type")
        if content_type:
            out_headers["content-type"] = content_type
        return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)

    async def _dispatch(self, backend_base: str, sem: asyncio.Semaphore, item: _QueuedRequest) -> None:
        acquired = False
        try:
            await sem.acquire()
            acquired = True
            if backend_base == BACKEND_A:
                self.inflight_a += 1
            else:
                self.inflight_b += 1

            response = await self._forward(backend_base, item)
            if not item.future.done():
                item.future.set_result(response)
        except Exception as e:
            if not item.future.done():
                item.future.set_result(JSONResponse(status_code=502, content={"detail": f"Upstream error: {e}"}))
        finally:
            if acquired:
                if backend_base == BACKEND_A:
                    self.inflight_a = max(0, self.inflight_a - 1)
                else:
                    self.inflight_b = max(0, self.inflight_b - 1)
                sem.release()

    async def _run(self) -> None:
        while not self._stop.is_set():
            item = await self.queue.get()
            # Drop requests that already timed out/cancelled at the handler.
            if item.future.done():
                continue
            pinned = self._pinned_backend(item.request.url.path)
            if pinned is not None:
                backend_base, sem = pinned
            else:
                backend_base, sem = self._next_backend()
            asyncio.create_task(self._dispatch(backend_base, sem, item))


STATE = _RouterState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await STATE.start()
    yield
    await STATE.stop()


app = FastAPI(title="Queued Router API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "router",
        "backend_a": BACKEND_A,
        "backend_b": BACKEND_B,
        "queue_depth": STATE.queue.qsize(),
        "queue_max": QUEUE_MAX,
        "inflight_a": STATE.inflight_a,
        "inflight_b": STATE.inflight_b,
        "capacity_a": BACKEND_A_CAPACITY,
        "capacity_b": BACKEND_B_CAPACITY,
        "uptime_sec": int(time.time() - STATE.started_at),
        "router_pin_backend": ROUTER_PIN_BACKEND or None,
        "router_pin_path_prefixes": list(ROUTER_PIN_PATH_PREFIXES),
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_all(path: str, request: Request):
    # Let router health be handled locally.
    if request.url.path == "/health":
        return await health()

    body = await request.body()
    return await STATE.enqueue(request, body)

