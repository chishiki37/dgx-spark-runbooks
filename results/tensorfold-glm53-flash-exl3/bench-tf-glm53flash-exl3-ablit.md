## tf-glm53flash-exl3-ablit (2026-10-09T15:08:16Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.36 | 70.64 | 0.324 |
| C4 | 129.88 | 41.5 | 0.525 |
| C8 | 123.48 | 38.29 | 2.905 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 77.17 | 44.25 | 42.46 |
| json | 83.72 | 43.86 | 44.19 |
| narrative | 44.22 | 23.39 | 22.31 |
| prose | 46.51 | 25.17 | 23.7 |
| math | 81.54 | 48.33 | 41.77 |
| reasoning | 60.24 | 32.43 | 29.5 |
| summary | 48.66 | 27.0 | 22.57 |
| structured | 101.51 | 67.37 | 60.75 |
| format | 92.17 | 61.72 | 57.33 |
| ceiling_count | 106.84 | 65.0 | 59.88 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding |  |  |  |
| json |  |  |  |
| narrative |  |  |  |
| prose |  |  |  |
| math |  |  |  |
| reasoning |  |  |  |
| summary |  |  |  |
| structured |  |  |  |
| format |  |  |  |
| ceiling_count |  |  |  |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1488 | 0.927 | 1604.6 |
| 8000 | 5932 | 3.333 | 1779.7 |
| 32000 | 23838 | 12.388 | 1924.3 |
