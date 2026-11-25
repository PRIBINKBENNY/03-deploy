"""Read vLLM's Prometheus /metrics endpoint: the server's own view of TTFT, inter-token latency, queueing
and KV-cache use. Compare it with what loadtest.py measured from the client side.

    python -m deploy.metrics --url http://localhost:8000/metrics
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict

import httpx

# vLLM has renamed some metrics across versions, so each entry lists accepted names.
HISTOGRAMS = {
    "ttft": ["vllm:time_to_first_token_seconds"],
    "inter_token_latency": ["vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"],
    "e2e_latency": ["vllm:e2e_request_latency_seconds"],
    "queue_time": ["vllm:request_queue_time_seconds"],
}
GAUGES = {
    "requests_running": ["vllm:num_requests_running"],
    "requests_waiting": ["vllm:num_requests_waiting"],
    "kv_cache_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
}
COUNTERS = {
    "prompt_tokens_total": ["vllm:prompt_tokens_total"],
    "generation_tokens_total": ["vllm:generation_tokens_total"],
}

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse(text: str) -> dict[str, list[tuple[dict, float]]]:
    samples: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        samples[name].append((dict(_LABEL.findall(labels or "")), float(value)))
    return samples


def _first(samples, names, suffix=""):
    for n in names:
        if n + suffix in samples:
            return n
    return None


def histogram_stats(samples, names) -> dict | None:
    base = _first(samples, names, "_count")
    if not base:
        return None
    count = sum(v for _, v in samples[base + "_count"])
    total = sum(v for _, v in samples[base + "_sum"])
    if count == 0:
        return {"count": 0}
    buckets = defaultdict(float)  # sum over label sets (e.g. several models)
    for labels, v in samples.get(base + "_bucket", []):
        le = labels.get("le")
        buckets[math.inf if le in ("+Inf", "inf") else float(le)] += v
    edges = sorted(buckets)

    def quantile(q: float) -> float:
        target = q * count
        prev_edge, prev_cum = 0.0, 0.0
        for e in edges:
            cum = buckets[e]
            if cum >= target:
                if math.isinf(e):
                    return prev_edge
                frac = (target - prev_cum) / (cum - prev_cum) if cum > prev_cum else 0
                return prev_edge + (e - prev_edge) * frac
            prev_edge, prev_cum = e, cum
        return prev_edge

    return {"count": int(count), "mean": round(total / count, 4),
            "p50": round(quantile(0.5), 4), "p90": round(quantile(0.9), 4), "p99": round(quantile(0.99), 4)}


def scalar(samples, names) -> float | None:
    n = _first(samples, names)
    return round(sum(v for _, v in samples[n]), 4) if n else None


def snapshot(text: str) -> dict:
    s = parse(text)
    out = {"histograms": {}, "gauges": {}, "counters": {}}
    for key, names in HISTOGRAMS.items():
        st = histogram_stats(s, names)
        if st is not None:
            out["histograms"][key] = st
    for key, names in GAUGES.items():
        out["gauges"][key] = scalar(s, names)
    for key, names in COUNTERS.items():
        out["counters"][key] = scalar(s, names)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/metrics")
    p.add_argument("--out")
    args = p.parse_args(argv)
    snap = snapshot(httpx.get(args.url, timeout=10).text)
    text = json.dumps(snap, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
