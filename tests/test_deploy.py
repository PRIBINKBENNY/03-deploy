"""Runs the real load tester against the mock vLLM-like server on a random local port."""
import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from deploy import loadtest, metrics, report
from deploy.mock_server import create_app


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server():
    port = free_port()
    cfg = uvicorn.Config(create_app(base_ttft=0.02, per_token=0.002), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=5)


def test_single_request_measures_ttft_and_itl(server):
    lr = asyncio.run(loadtest.run_level(server + "/v1", "m", 1, 3, 20, ["hi there"], "k", 30))
    assert lr.errors == 0 and lr.output_tokens == 60
    assert 0.015 < lr.ttft_s["p50"] < 0.2
    assert 0.001 < lr.itl_s["p50"] < 0.02
    assert lr.latency_s["p50"] > lr.ttft_s["p50"]


def test_concurrency_raises_throughput(server):
    one = asyncio.run(loadtest.run_level(server + "/v1", "m", 1, 8, 30, ["x"], "k", 30))
    many = asyncio.run(loadtest.run_level(server + "/v1", "m", 8, 8, 30, ["x"], "k", 30))
    assert many.output_throughput_tok_s > 3 * one.output_throughput_tok_s
    assert many.ttft_s["p50"] > one.ttft_s["p50"]  # queueing shows up in TTFT


def test_bad_endpoint_counts_errors_not_crash():
    lr = asyncio.run(loadtest.run_level("http://127.0.0.1:9/v1", "m", 2, 4, 5, ["x"], "k", 2))
    assert lr.errors == 4 and lr.output_throughput_tok_s == 0


def test_metrics_parser_reads_vllm_histograms(server):
    asyncio.run(loadtest.run_level(server + "/v1", "m", 4, 8, 10, ["x"], "k", 30))
    snap = metrics.snapshot(httpx.get(server + "/metrics").text)
    h = snap["histograms"]
    assert h["ttft"]["count"] >= 8 and 0 < h["ttft"]["p50"] < 1
    assert h["inter_token_latency"]["count"] > 0
    assert snap["counters"]["generation_tokens_total"] >= 80


def test_metrics_parser_handles_old_names_and_quantiles():
    text = """
# old vLLM name
vllm:time_per_output_token_seconds_bucket{le="0.01"} 50
vllm:time_per_output_token_seconds_bucket{le="0.02"} 90
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 100
vllm:time_per_output_token_seconds_sum 1.5
vllm:time_per_output_token_seconds_count 100
vllm:gpu_cache_usage_perc{model_name="x"} 0.42
"""
    snap = metrics.snapshot(text)
    itl = snap["histograms"]["inter_token_latency"]
    assert itl["mean"] == 0.015 and itl["p50"] == pytest.approx(0.01)
    assert 0.01 < itl["p90"] <= 0.02
    assert snap["gauges"]["kv_cache_usage"] == 0.42


def test_report_writes_table_into_readme(tmp_path, server):
    out = tmp_path / "loadtest.json"
    rc = loadtest.main(["--base-url", server + "/v1", "--model", "m", "--concurrency", "1", "4",
                        "--requests", "4", "--max-tokens", "16", "--out", str(out), "--label", "pytest mock"])
    assert rc == 0
    readme = tmp_path / "README.md"
    readme.write_text("# x\n\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n\nfooter\n")
    report.main(["--results", str(out), "--readme", str(readme)])
    text = readme.read_text()
    assert "old" not in text and "footer" in text and "pytest mock" in text
    assert "| 4 | 4 | 0 |" in text
    assert json.loads(out.read_text())["levels"][1]["concurrency"] == 4
