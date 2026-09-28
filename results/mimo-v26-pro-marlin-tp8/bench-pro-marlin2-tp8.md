## pro-marlin2-tp8 (2026-09-28T02:08:56Z)

TP8 DFlash-7 marlin linear+EP full recipe flags (ep-weight-filter, FULL_DECODE_ONLY cudagraph, vllm_c rmsnorm, draft TP8) A/B

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 31.43 | 35.89 | 0.471 |
| C4 | 62.69 | 19.05 | 0.921 |
| C8 | 78.74 | 11.82 | 1.418 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 49.39 | 30.69 | 15.77 |
| json | 39.04 | 21.4 | 12.59 |
| narrative | 17.61 | 7.4 | 4.68 |
| prose | 18.14 | 8.41 | 6.03 |
| math | 38.31 | 22.95 | 14.1 |
| reasoning | 20.99 | 15.63 | 8.41 |
| summary | 19.14 | 10.79 | 6.72 |
| structured | 54.37 | 27.31 | 19.59 |
| format | 66.0 | 26.87 | 18.49 |
| ceiling_count | 75.07 | 31.05 | 22.13 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 5.19 | 4.78 | 4.96 |
| json | 3.61 | 3.07 | 3.42 |
| narrative | 0.74 | 0.78 | 0.71 |
| prose | 1.2 | 1.01 | 1.21 |
| math | 4.25 | 4.07 | 4.28 |
| reasoning | 2.01 | 2.11 | 2.32 |
| summary | 1.69 | 1.58 | 1.65 |
| structured | 6.14 | 6.2 | 6.09 |
| format | 5.94 | 5.91 | 5.5 |
| ceiling_count | 6.88 | 6.92 | 6.88 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2004 | 1.815 | 1104.2 |
| 8000 | 7919 | 14.262 | 555.3 |
| 32000 | 31836 | 57.368 | 554.9 |
