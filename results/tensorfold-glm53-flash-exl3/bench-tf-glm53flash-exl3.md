## tf-glm53flash-exl3 (2026-10-09T11:32:27Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.39 | 68.56 | 0.307 |
| C4 | 116.44 | 35.88 | 0.695 |
| C8 | 119.94 | 35.6 | 3.14 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 84.29 | 49.09 | 46.04 |
| json | 80.08 | 43.24 | 37.05 |
| narrative | 45.26 | 23.43 | 22.14 |
| prose | 50.97 | 24.47 | 23.75 |
| math | 78.94 | 43.76 | 42.56 |
| reasoning | 67.93 | 28.9 | 31.23 |
| summary | 48.12 | 27.68 | 24.38 |
| format | 92.9 | 46.49 | 57.68 |
| ceiling_count | 105.36 | 58.66 | 59.64 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 3807 | 3.859 | 986.4 |
| 8000 | 15161 | 7.95 | 1907.1 |
| 32000 | 60910 | 32.456 | 1876.7 |
