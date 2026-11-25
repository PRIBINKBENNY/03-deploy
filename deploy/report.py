"""Turn results/loadtest.json (+ optional metrics snapshot) into a results table and charts,
and write them into README.md between the RESULTS markers.

    python -m deploy.report --results results/loadtest.json --metrics results/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def ms(d: dict, key: str) -> str:
    v = d.get(key)
    return "–" if v is None else f"{v * 1000:.0f}"


def table(res: dict) -> str:
    rows = [
        "| concurrency | requests | errors | output throughput (tok/s) | req/s | TTFT p50 / p90 (ms) "
        "| inter-token p50 / p90 (ms) | e2e latency p50 / p90 (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for l in res["levels"]:
        lat = l["latency_s"]
        rows.append(
            f"| {l['concurrency']} | {l['requests']} | {l['errors']} | **{l['output_throughput_tok_s']:.0f}** "
            f"| {l['request_rate_rps']:.2f} | {ms(l['ttft_s'], 'p50')} / {ms(l['ttft_s'], 'p90')} "
            f"| {ms(l['itl_s'], 'p50')} / {ms(l['itl_s'], 'p90')} "
            f"| {lat.get('p50', 0):.2f} / {lat.get('p90', 0):.2f} |"
        )
    return "\n".join(rows)


def metrics_table(m: dict) -> str:
    rows = ["| server metric (vLLM /metrics) | mean | p50 | p90 | p99 |", "|---|---:|---:|---:|---:|"]
    names = {"ttft": "time to first token (s)", "inter_token_latency": "inter-token latency (s)",
             "e2e_latency": "end-to-end latency (s)", "queue_time": "queue time (s)"}
    for k, label in names.items():
        h = m.get("histograms", {}).get(k)
        if h and h.get("count"):
            rows.append(f"| {label} | {h['mean']:.3f} | {h['p50']:.3f} | {h['p90']:.3f} | {h['p99']:.3f} |")
    return "\n".join(rows)


def charts(res: dict, out_dir: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    c = [l["concurrency"] for l in res["levels"]]
    thr = [l["output_throughput_tok_s"] for l in res["levels"]]
    ttft = [l["ttft_s"].get("p50", 0) * 1000 for l in res["levels"]]
    itl = [l["itl_s"].get("p50", 0) * 1000 for l in res["levels"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(c, thr, marker="o", color="#2a6fdb")
    axes[0].set(title="Output throughput", xlabel="concurrent requests", ylabel="tokens / s")
    axes[1].plot(c, ttft, marker="o", label="TTFT p50", color="#d9822b")
    axes[1].plot(c, itl, marker="s", label="inter-token p50", color="#3a9d5d")
    axes[1].set(title="Latency", xlabel="concurrent requests", ylabel="ms")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(c, [str(x) for x in c])
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "loadtest.png"
    fig.savefig(path, dpi=150)
    return path.as_posix()


def build(res: dict, metrics: dict | None, chart: str | None) -> str:
    best = max(res["levels"], key=lambda l: l["output_throughput_tok_s"])
    single = min(res["levels"], key=lambda l: l["concurrency"])
    parts = [
        f"**Setup:** {res.get('label') or 'unlabelled run'} · model `{res['model']}` · "
        f"max {res['max_tokens']} output tokens · {res['timestamp']}",
        "",
        f"- **TTFT (1 user):** {ms(single['ttft_s'], 'p50')} ms p50",
        f"- **Inter-token latency (1 user):** {ms(single['itl_s'], 'p50')} ms p50 "
        f"(≈ {1 / single['itl_s']['p50']:.0f} tok/s per stream)" if single["itl_s"].get("p50") else "",
        f"- **Peak throughput:** {best['output_throughput_tok_s']:.0f} tok/s at {best['concurrency']} concurrent requests",
        "",
        table(res),
    ]
    if chart:
        parts += ["", f"![load test]({chart})"]
    if metrics:
        parts += ["", metrics_table(metrics)]
    return "\n".join(p for p in parts if p is not None)


def write_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    if START in text and END in text:
        pre, _, rest = text.partition(START)
        _, _, post = rest.partition(END)
        text = f"{pre}{START}\n{block}\n{END}{post}"
    else:
        text += f"\n\n## Results\n\n{START}\n{block}\n{END}\n"
    readme.write_text(text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/loadtest.json")
    p.add_argument("--metrics", help="JSON from `python -m deploy.metrics --out ...`")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--no-readme", action="store_true", help="print only")
    args = p.parse_args(argv)
    res = json.loads(Path(args.results).read_text())
    metrics = json.loads(Path(args.metrics).read_text()) if args.metrics else None
    chart = charts(res, Path(args.results).parent)
    block = build(res, metrics, chart)
    print(block)
    if not args.no_readme:
        write_readme(Path(args.readme), block)
        print(f"\nupdated {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
