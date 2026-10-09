## ours9 (2026-09-10T16:47:13Z)

fleet reproduction, NCCL cross_nic fix, image ours-a5

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 31.67 | 36.22 | 0.556 |
| C2 | 54.34 | 31.81 | 0.768 |
| C3 | 64.71 | 24.96 | 0.567 |
| C4 | 75.24 | 21.64 | 0.665 |
| C5 | 88.02 | 20.41 | 0.677 |
| C6 | 99.19 | 19.19 | 0.784 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 48.29 | 41.68 | 36.1 | 29.23 | 29.02 | 26.83 |
| json | 30.86 | 25.81 | 24.9 | 19.36 | 17.76 | 19.06 |
| narrative | 20.19 | 14.02 | 12.2 | 9.8 | 8.67 | 8.26 |
| prose | 23.09 | 23.13 | 15.0 | 12.28 | 10.73 | 9.88 |
| math | 47.89 | 55.43 | 33.95 | 30.48 | 27.76 | 27.83 |
| reasoning | 39.78 | 29.2 | 21.98 | 21.43 | 20.21 | 18.28 |
| summary | 24.99 | 16.33 | 15.73 | 12.77 | 17.52 | 10.13 |
| format | 54.64 | 48.85 | 39.78 | 37.75 | 31.63 | 33.24 |
| ceiling_count | 56.93 | 52.0 | 42.68 | 41.32 | 36.7 | 36.9 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 6.302 | 468.1 |
| 8000 | 11592 | 31.394 | 369.2 |
| 32000 | 46810 | 87.05 | 537.7 |
| 64000 | 93335 | 151.869 | 614.6 |
