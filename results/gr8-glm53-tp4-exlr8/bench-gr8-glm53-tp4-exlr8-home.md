## gr8-glm53-tp4-exlr8-home (2026-10-17T12:29:45Z)

gr8 v0.5 recipe (knapcio) + EXLR8 home manifest, 4x Spark TP4, RoCE single rail, reasoning_effort low (no thinking-off mode), v1 prompt set

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (9 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 47.4 | 55.95 | 0.492 |
| C4 | 63.41 | 22.3 | 1.927 |
| C8 | 74.5 | 15.47 | 4.191 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 67.73 | 24.54 | 16.8 |
| json | 56.08 | 24.52 | 17.68 |
| narrative | 36.6 | 16.86 | 11.76 |
| prose | 42.64 | 21.21 | 13.82 |
| math | 68.43 | 24.43 | 17.0 |
| reasoning | 40.49 | 19.53 | 12.89 |
| summary | 45.64 | 19.21 | 12.88 |
| structured | 78.05 | 23.47 | 18.26 |
| format | 67.85 | 26.89 | 18.16 |
| ceiling_count | 82.96 | 26.94 | 19.13 |

### DFlash accepted tokens per draft step (7 drafted per step)

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 5.03 | 1.98 | 1.96 |
| json | 3.52 | 1.8 | 1.78 |
| narrative | 0.93 | 0.91 | 0.91 |
| prose | 1.52 | 1.35 | 1.24 |
| math | 4.47 | 1.8 | 1.8 |
| reasoning | 1.39 | 1.3 | 1.2 |
| summary | 1.71 | 1.37 | 1.37 |
| structured | 6.07 | 1.87 | 1.97 |
| format | 4.29 | 2.02 | 1.88 |
| ceiling_count | 6.52 | 2.03 | 2.0 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 2.125 | 702.9 |
| 8000 | 5938 | 6.343 | 936.2 |
| 32000 | 23844 | 20.828 | 1144.8 |
| 64000 | 47771 | 42.868 | 1114.4 |
