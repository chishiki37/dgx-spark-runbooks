## tf-flashnext (2026-10-05T15:31:16Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 63.43 | 69.98 | 0.258 |
| C4 | 129.82 | 37.73 | 0.451 |
| C8 | 152.44 | 41.17 | 2.437 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 84.79 | 24.42 | 40.41 |
| json | 83.81 | 49.07 | 58.1 |
| narrative | 47.99 | 24.18 | 24.01 |
| prose | 46.9 | 30.45 | 28.87 |
| math | 85.42 | 44.82 | 44.03 |
| reasoning | 73.67 | 45.26 | 47.72 |
| summary | 42.3 | 26.16 | 24.69 |
| format | 94.99 | 57.47 | 61.56 |
| ceiling_count | 115.0 | 88.28 | 85.98 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5075 | 2.988 | 1698.6 |
| 8000 | 20117 | 9.243 | 2176.6 |
| 32000 | 80629 | 40.635 | 1984.2 |
