## sglang-a14 (2026-09-28T06:06:20Z)

rhys101 a14 SGLang recipe, single-rail NCCL (RoCEnante off), all other flags at their defaults

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 46.7 | 54.3 | 0.406 |
| C4 | 110.96 | 34.23 | 0.716 |
| C8 | 157.44 | 24.05 | 0.942 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 55.87 | 43.99 | 31.37 |
| json | 67.82 | 37.11 | 27.77 |
| narrative | 29.23 | 18.42 | 10.55 |
| prose | 35.51 | 18.89 | 14.27 |
| math | 73.05 | 46.78 | 31.1 |
| reasoning | 41.45 | 22.12 | 16.56 |
| summary | 37.3 | 25.09 | 16.12 |
| format | 94.15 | 61.4 | 44.69 |
| ceiling_count | 108.51 | 71.46 | 51.61 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5115 | 6.082 | 841.0 |
| 8000 | 20264 | 17.326 | 1169.6 |
| 32000 | 81292 | 70.155 | 1158.8 |
