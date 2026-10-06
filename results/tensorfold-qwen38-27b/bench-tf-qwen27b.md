## tf-qwen27b (2026-10-06T00:42:25Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 80.11 | 85.37 | 0.133 |
| C4 | 184.38 | 57.2 | 0.495 |
| C8 | 243.92 | 41.35 | 0.983 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 114.31 | 86.82 | 56.39 |
| json | 111.0 | 75.44 | 52.33 |
| narrative | 40.86 | 27.77 | 22.31 |
| prose | 43.22 | 34.11 | 25.39 |
| math | 114.98 | 78.98 | 54.06 |
| reasoning | 81.31 | 54.43 | 40.65 |
| summary | 49.82 | 31.71 | 24.01 |
| format | 127.49 | 68.33 | 55.69 |
| ceiling_count | 179.3 | 121.75 | 76.74 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5075 | 2.72 | 1865.8 |
| 8000 | 20117 | 11.213 | 1794.1 |
| 32000 | 80629 | 57.915 | 1392.2 |
