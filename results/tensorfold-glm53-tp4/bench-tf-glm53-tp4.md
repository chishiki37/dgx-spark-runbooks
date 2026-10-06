## tf-glm53-tp4 (2026-10-06T09:27:19Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 37.11 | 40.68 | 0.411 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 43.94 |
| json | 41.9 |
| narrative | 35.62 |
| prose | 37.95 |
| math | 40.53 |
| reasoning | 43.33 |
| summary | 38.36 |
| format | 43.81 |
| ceiling_count | 51.65 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 3814 | 3.446 | 1106.9 |
| 8000 | 15168 | 13.803 | 1098.9 |
| 32000 | 60917 | 57.931 | 1051.5 |
