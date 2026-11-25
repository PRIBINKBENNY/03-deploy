# 03 · Deploy an open-source model

Serve an open model behind an **OpenAI-compatible API** on a **free GPU**, **load-test** it, and report **TTFT, inter-token latency and throughput**.

```
$ pip install vllm
$ vllm serve unsloth/Llama-3.2-1B-Instruct      # or: python -m sglang.launch_server
$ curl :8000/metrics
time_to_first_token
inter_token_latency
throughput   # tokens per second
```

## Run it (free Colab T4, ~10 min)

Open [`notebooks/colab_deploy.ipynb`](notebooks/colab_deploy.ipynb) in Colab, pick **Runtime → T4 GPU**, and run all cells. The notebook:

1. starts vLLM (`--dtype half`: the T4 has no bf16; `--max-model-len 4096`)
2. checks the endpoint with a real chat request
3. runs the load test at concurrency 1 / 4 / 16 / 32
4. snapshots vLLM's `/metrics` (the server-side view)
5. writes the results table and chart into this README
6. *(optional)* opens a Cloudflare tunnel so **01-rag** and **02-agents** can use your model

On your own GPU box:

```bash
pip install -e ".[serve]"
./serve.sh                                   # MODEL=Qwen/Qwen3-4B ./serve.sh, ENGINE=sglang ./serve.sh
python -m deploy.loadtest --concurrency 1 4 16 32 --requests 64 --label "RTX 4090 · vLLM · fp16"
python -m deploy.metrics --out results/metrics.json
python -m deploy.report --metrics results/metrics.json   # updates the Results section below
```

## What gets measured

| metric | meaning | measured by |
|---|---|---|
| **TTFT** | request sent → first token. Dominated by queueing + prefill | client (`loadtest.py`) and server (`vllm:time_to_first_token_seconds`) |
| **inter-token latency** | gap between later tokens, i.e. how fast text streams | client and `vllm:inter_token_latency_seconds` |
| **e2e latency** | request → last token | client and `vllm:e2e_request_latency_seconds` |
| **output throughput** | generated tokens / second across all users, i.e. capacity | client |

The load tester streams real requests (`stream: true`, `include_usage`) with an async semaphore per concurrency level and reports mean/p50/p90/p99. `metrics.py` parses vLLM's Prometheus histograms (new and old metric names) and estimates quantiles from the buckets.

Expect TTFT to rise with concurrency (requests queue for prefill) while throughput climbs, because vLLM batches decode steps, until the GPU saturates. The knee of that curve is the number to put in a capacity plan.

## Results

<!-- RESULTS:START -->
*Run the notebook on a T4; `deploy.report` replaces this block with your TTFT, latency and throughput table plus `results/loadtest.png`.*

A dry run of the whole pipeline against the bundled mock server is in [`results/mock_dryrun.md`](results/mock_dryrun.md). It shows the format only; those are **not** GPU numbers.
<!-- RESULTS:END -->

## Which model fits a free T4 (~15 GB)?

| model | size | fp16 memory | free T4? |
|---|---|---|---|
| Llama-3.2-1B-Instruct | 1B | ~2.5 GB | yes, start here |
| Qwen3-4B | 4B | ~8 GB | yes |
| any 8B model | 8B | ~16 GB | use a 4-bit AWQ checkpoint (`--quantization awq`) |

Rule of thumb: ~2 GB per billion parameters in fp16, plus room for the **KV cache**, which grows with context length × concurrent users. `--gpu-memory-utilization 0.9` tells vLLM to reserve 90% of the card for weights + KV cache; `--max-model-len` caps how much any one request can take.

Free GPUs: **Google Colab** (T4, sessions up to ~12 h, not always available at peak), **Lightning AI** (free Studio plus monthly GPU credits).

## Test without a GPU

`deploy/mock_server.py` is a fake OpenAI-compatible server that behaves roughly like vLLM under load and exposes vLLM-style `/metrics`. The tests run the real load tester, metrics parser and report writer against it:

```bash
pip install -e ".[dev]"
pytest                                  # 6 tests
python -m deploy.mock_server &          # or poke at it by hand
python -m deploy.loadtest --model mock --concurrency 1 4 16
```

## Layout

```
serve.sh                     vLLM / SGLang launch with T4-safe defaults
notebooks/colab_deploy.ipynb end-to-end run on a free GPU
deploy/loadtest.py           async streaming load tester: TTFT, ITL, latency, throughput
deploy/metrics.py            vLLM /metrics (Prometheus) parser
deploy/report.py             results table + chart -> README
deploy/mock_server.py        GPU-free stand-in for tests
```

## What I learned

vLLM · SGLang · GPU memory · TTFT · inter-token latency · throughput

Start here: vLLM docs, Production Metrics (free): docs.vllm.ai/en/latest/usage/metrics.html
