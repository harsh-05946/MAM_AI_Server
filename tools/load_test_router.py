#!/usr/bin/env python3
"""
Very small load test for the RouterAPI.

Example:
  python tools/load_test_router.py --url http://127.0.0.1:9000/health --concurrency 50 --requests 500

If you want to hit an inference endpoint, pass --method POST and --json-body '{"texts":["hello"]}'
to /process/embeddings (or similar).
"""

import argparse
import asyncio
import json
import time
from statistics import mean

import httpx


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((len(s) - 1) * p))
    return s[max(0, min(k, len(s) - 1))]


async def worker(name: str, client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str, method: str, body: bytes | None, headers: dict[str, str], latencies: list[float], errors: list[str], n: int):
    for _ in range(n):
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await client.request(method=method, url=url, content=body, headers=headers)
                if resp.status_code >= 400:
                    errors.append(f"{resp.status_code}:{resp.text[:200]}")
            except Exception as e:
                errors.append(str(e))
            finally:
                latencies.append((time.perf_counter() - t0) * 1000.0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9000/health")
    ap.add_argument("--method", default="GET")
    ap.add_argument("--concurrency", type=int, default=25)
    ap.add_argument("--requests", type=int, default=250)
    ap.add_argument("--json-body", default="")
    args = ap.parse_args()

    method = args.method.upper().strip()
    body = None
    headers: dict[str, str] = {}
    if args.json_body:
        parsed = json.loads(args.json_body)
        body = json.dumps(parsed).encode("utf-8")
        headers["content-type"] = "application/json"

    sem = asyncio.Semaphore(max(1, args.concurrency))
    per_worker = max(1, args.requests // max(1, args.concurrency))

    latencies: list[float] = []
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(worker(f"w{i}", client, sem, args.url, method, body, headers, latencies, errors, per_worker))
            for i in range(max(1, args.concurrency))
        ]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    ok = max(0, len(latencies) - len(errors))
    print(json.dumps({
        "url": args.url,
        "method": method,
        "sent": len(latencies),
        "ok": ok,
        "errors": len(errors),
        "elapsed_sec": round(elapsed, 3),
        "rps": round(len(latencies) / max(elapsed, 1e-6), 2),
        "lat_ms_avg": round(mean(latencies), 2) if latencies else 0.0,
        "lat_ms_p50": round(_pct(latencies, 0.50), 2),
        "lat_ms_p95": round(_pct(latencies, 0.95), 2),
        "lat_ms_p99": round(_pct(latencies, 0.99), 2),
        "sample_error": errors[0] if errors else "",
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

