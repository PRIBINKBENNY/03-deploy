"""Load-test an OpenAI-compatible endpoint (vLLM, SGLang, ...) and measure what users feel.

    python -m deploy.loadtest --base-url http://localhost:8000/v1 --model unsloth/Llama-3.2-1B-Instruct \
        --concurrency 1 4 16 --requests 32 --max-tokens 128

Per request (streaming):
  TTFT              time from sending the request to the first generated token
  inter-token lat.  average gap between later tokens (how smooth the stream feels)
  e2e latency       time until the last token
Per concurrency level:
  output throughput total generated tokens / wall-clock seconds (the server's capacity)
  p50 / p90 / p99 of the above, request rate, error count
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

DEFAULT_PROMPTS = [
    "Explain what a KV cache is in two short paragraphs.",
    "Write a haiku about GPUs, then explain it.",
    "List five tips for writing clean Python code.",
    "Summarise the plot of a heist movie you invent, in about 100 words.",
    "What is the difference between latency and throughput? Give an example.",
    "Describe how a robot arm uses inverse kinematics, simply.",
    "Give a step-by-step recipe for masala chai.",
    "Explain retrieval-augmented generation to a new engineer.",
]


@dataclass
class RequestResult:
    ok: bool
    ttft: float | None = None
    itl: float | None = None  # mean inter-token latency for this request
    latency: float | None = None
    output_tokens: int = 0
    error: str = ""


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    errors: int
    wall_s: float
    output_tokens: int
    output_throughput_tok_s: float
    request_rate_rps: float
    ttft_s: dict = field(default_factory=dict)
    itl_s: dict = field(default_factory=dict)
    latency_s: dict = field(default_factory=dict)
    tokens_per_request: float = 0.0


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summary(values: list[float]) -> dict:
    if not values:
        return {}
    return {"mean": round(statistics.fmean(values), 4), "p50": round(pct(values, 50), 4),
            "p90": round(pct(values, 90), 4), "p99": round(pct(values, 99), 4)}


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int,
                      api_key: str) -> RequestResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.perf_counter()
    token_times: list[float] = []
    usage_tokens = None
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors="ignore")[:200]
                return RequestResult(False, error=f"HTTP {resp.status_code}: {text}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens")
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        token_times.append(time.perf_counter())
    except Exception as e:  # network errors count as failed requests, they don't crash the test
        return RequestResult(False, error=f"{type(e).__name__}: {e}")

    end = time.perf_counter()
    if not token_times:
        return RequestResult(False, error="no tokens received")
    ttft = token_times[0] - t0
    n_out = usage_tokens or len(token_times)
    # each streamed chunk is usually one token; spread the decode time over the tokens after the first
    itl = (token_times[-1] - token_times[0]) / (n_out - 1) if n_out > 1 else None
    return RequestResult(True, ttft, itl, end - t0, n_out)


async def run_level(base_url: str, model: str, concurrency: int, n_requests: int, max_tokens: int,
                    prompts: list[str], api_key: str, timeout: float) -> LevelResult:
    url = base_url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def worker(i: int) -> RequestResult:
            async with sem:
                return await one_request(client, url, model, prompts[i % len(prompts)], max_tokens, api_key)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(worker(i) for i in range(n_requests)))
        wall = time.perf_counter() - t0

    good = [r for r in results if r.ok]
    out_tokens = sum(r.output_tokens for r in good)
    return LevelResult(
        concurrency=concurrency,
        requests=n_requests,
        errors=n_requests - len(good),
        wall_s=round(wall, 3),
        output_tokens=out_tokens,
        output_throughput_tok_s=round(out_tokens / wall, 2) if wall else 0.0,
        request_rate_rps=round(len(good) / wall, 3) if wall else 0.0,
        ttft_s=summary([r.ttft for r in good if r.ttft is not None]),
        itl_s=summary([r.itl for r in good if r.itl is not None]),
        latency_s=summary([r.latency for r in good if r.latency is not None]),
        tokens_per_request=round(out_tokens / len(good), 1) if good else 0.0,
    )


async def warmup(base_url: str, model: str, api_key: str) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        await one_request(client, base_url.rstrip("/") + "/chat/completions", model, "Say hi.", 8, api_key)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--api-key", default="not-needed")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--requests", type=int, default=32, help="requests per concurrency level")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompts", help="text file, one prompt per line")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--out", default="results/loadtest.json")
    p.add_argument("--label", default="", help="e.g. 'Colab T4, vLLM 0.x, fp16'")
    args = p.parse_args(argv)

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [l.strip() for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    random.Random(0).shuffle(prompts)

    asyncio.run(warmup(args.base_url, args.model, args.api_key))
    levels = []
    for c in args.concurrency:
        n = max(args.requests, c)
        print(f"concurrency {c:>3}: {n} requests ...", flush=True)
        lr = asyncio.run(run_level(args.base_url, args.model, c, n, args.max_tokens, prompts, args.api_key,
                                   args.timeout))
        levels.append(lr)
        print(f"   throughput {lr.output_throughput_tok_s:>8.1f} tok/s | TTFT p50 {lr.ttft_s.get('p50', float('nan')) * 1000:7.1f} ms"
              f" | ITL p50 {lr.itl_s.get('p50', float('nan')) * 1000:6.1f} ms | errors {lr.errors}")

    out = {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "levels": [asdict(l) for l in levels],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out}")
    return 0 if all(l.errors == 0 for l in levels) else 1


if __name__ == "__main__":
    raise SystemExit(main())
