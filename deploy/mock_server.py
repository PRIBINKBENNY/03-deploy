"""A fake OpenAI-compatible LLM server that behaves roughly like vLLM under load: TTFT grows with
queued prefills, per-token latency grows slowly with batch size, so throughput rises with concurrency
and then flattens. It exposes /metrics in vLLM's Prometheus format.

Use it to test the load tester and report pipeline without a GPU. Its numbers are NOT real benchmarks.

    python -m deploy.mock_server --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

MODEL = "mock/llama-3.2-1b"
BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]


class Histogram:
    def __init__(self):
        self.counts = [0] * len(BUCKETS)
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float):
        self.sum += v
        self.n += 1
        for i, b in enumerate(BUCKETS):
            if v <= b:
                self.counts[i] += 1

    def render(self, name: str) -> str:
        lines = [f"# TYPE {name} histogram"]
        for b, c in zip(BUCKETS, self.counts):
            lines.append(f'{name}_bucket{{le="{b}",model_name="{MODEL}"}} {c}')
        lines.append(f'{name}_bucket{{le="+Inf",model_name="{MODEL}"}} {self.n}')
        lines.append(f'{name}_sum{{model_name="{MODEL}"}} {self.sum}')
        lines.append(f'{name}_count{{model_name="{MODEL}"}} {self.n}')
        return "\n".join(lines)


def create_app(base_ttft: float = 0.04, per_token: float = 0.008, batch_penalty: float = 0.04) -> FastAPI:
    app = FastAPI()
    state = {"running": 0, "gen_tokens": 0, "prompt_tokens": 0}
    h = {"ttft": Histogram(), "itl": Histogram(), "e2e": Histogram()}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    @app.get("/metrics")
    async def metrics():
        parts = [
            h["ttft"].render("vllm:time_to_first_token_seconds"),
            h["itl"].render("vllm:inter_token_latency_seconds"),
            h["e2e"].render("vllm:e2e_request_latency_seconds"),
            f'vllm:num_requests_running{{model_name="{MODEL}"}} {state["running"]}',
            f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0',
            f'vllm:kv_cache_usage_perc{{model_name="{MODEL}"}} {min(1.0, state["running"] * 0.03):.3f}',
            f'vllm:prompt_tokens_total{{model_name="{MODEL}"}} {state["prompt_tokens"]}',
            f'vllm:generation_tokens_total{{model_name="{MODEL}"}} {state["gen_tokens"]}',
        ]
        return PlainTextResponse("\n".join(parts) + "\n")

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        max_tokens = int(body.get("max_tokens") or 64)
        prompt = " ".join(m.get("content", "") for m in body.get("messages", []))
        n_prompt = max(1, len(prompt.split()))
        include_usage = (body.get("stream_options") or {}).get("include_usage", False)

        if not body.get("stream"):
            text = " ".join(f"tok{i}" for i in range(max_tokens))
            return JSONResponse({"id": "mock", "object": "chat.completion", "model": MODEL,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                              "finish_reason": "length"}],
                                 "usage": {"prompt_tokens": n_prompt, "completion_tokens": max_tokens}})

        async def gen():
            t0 = time.perf_counter()
            state["running"] += 1
            state["prompt_tokens"] += n_prompt
            try:
                await asyncio.sleep(base_ttft * (1 + 0.5 * (state["running"] - 1)))  # prefill contention
                h["ttft"].observe(time.perf_counter() - t0)
                last = time.perf_counter()
                for i in range(max_tokens):
                    if i:
                        await asyncio.sleep(per_token * (1 + batch_penalty * state["running"]))
                        now = time.perf_counter()
                        h["itl"].observe(now - last)
                        last = now
                    state["gen_tokens"] += 1
                    chunk = {"id": "mock", "object": "chat.completion.chunk", "model": MODEL,
                             "choices": [{"index": 0, "delta": {"content": f" tok{i}"}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {"id": "mock", "object": "chat.completion.chunk", "model": MODEL,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
                yield f"data: {json.dumps(final)}\n\n"
                if include_usage:
                    usage = {"id": "mock", "object": "chat.completion.chunk", "model": MODEL, "choices": [],
                             "usage": {"prompt_tokens": n_prompt, "completion_tokens": max_tokens}}
                    yield f"data: {json.dumps(usage)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                state["running"] -= 1
                h["e2e"].observe(time.perf_counter() - t0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
