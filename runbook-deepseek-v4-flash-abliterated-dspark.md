# Runbook: DeepSeek V4 Flash Abliterated (drowzeys, NVFP4) on 2× DGX Spark

**Model:** `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` (GATED — HF token + author approval required)
**Hardware:** 2× NVIDIA DGX Spark (GB10, 128 GB unified)
**Architecture:** `DeepseekV4ForCausalLM` — 256 routed + 1 shared experts, 6 active/token, 43 layers, 1M max_position_embeddings
**Quantization:** NVFP4 (expert_dtype fp4) + FP8 attention — ~156 GB, 48 safetensors shards (NOT 384 GB despite shard count)
**Recommended runtime:** vLLM via the DSpark Docker image — same stack as the 0731 deployment
**Verified:** Aug 2, 2026

---

## TL;DR — how to serve it

Use the **same vLLM DSpark runtime** as the official 0731 model:

- Image: `vllm-dspark-runtime:dspark-nvfp4-stage-c` (the HF model README names `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, which is what this image is pre-installed as locally)
- Deployment dir / scripts: `~/deepseek-v4-flash-dspark` (see `runbook-deepseek-v4-flash-0731-dspark.md` for the full compose/env/launch procedure)
- Point `DSPARK_MODEL` at the abliterated repo instead of `deepseek-ai/DeepSeek-V4-Flash-0731`, keep revision-pinned download, 1M ctx, k=5 MTP, `VLLM_USE_BREAKABLE_CUDAGRAPH=0` — all findings from the 0731 runbook carry over (same arch, same DSpark draft heads).

**Always read the model README's serve section first** — it states the intended runtime. For this model it's the Anemll DSpark vLLM image, not the bundled custom code.

## ⚠️ DO NOT use the bundled custom inference code (on GB10)

The repo ships `inference/` (tilelang-based `convert.py` + `generate.py` + torchrun 2-node server). We ran it end-to-end (Aug 2, 2026) — it works but is a dead end:

- **Speed: ~1–4 tok/s** (vs 68+ tok/s on the vLLM path) — no batching, single-request generation lock
- Required to get there: OOM-safe streaming loader rewrite (meta-device + per-tensor `safe_open` copy; stock loader SIGKILLs at ~85 GB on 128 GB unified), re-registration of `.scale` FP4/FP8 attributes, tilelang 0.1.12 kernel patches (fp8_gemm stages 4→3, sparse_attn block 64→32 — GB10 sm_121 dyn-smem limit is 101,376 bytes), pure-PyTorch Hadamard replacement (fast-hadamard-transform won't build), per-request `torch.set_default_device("cuda")` (thread-local default device breaks under FastAPI threadpool), NCCL-safe streaming via background-thread + queue (yielding between forward() calls desyncs ranks → SIGABRT).
- Patched copies live in `~/deepseek-inference/` on both nodes if you ever need them.

**Verdict:** custom inference was validated as functional, then abandoned for vLLM. Keep it only as a reference for tilelang/GB10 smem limits.

## Download (gated)

```python
from huggingface_hub import snapshot_download
snapshot_download("drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32")
# default cache path only — honors HF_HOME → ~/.cache/huggingface/hub/
```

Requires an HF token that has been granted access to the gated repo. Watch the same `hub/` vs non-`hub/` cache-dir trap as the 0731 runbook.

## NFS sharing (if only one node holds the weights)

```bash
# On model host (e.g. bdea)
echo '/path/to/model 10.10.10.0/24(ro,sync,no_subtree_check)' | sudo tee -a /etc/exports
sudo exportfs -ra
# On worker (e.g. 9105) — RDMA transport
sudo mount -t nfs -o vers=3,proto=rdma,port=20049,ro <bdea-fabric-ip>:/path/to/model /mnt/deepseek-nfs
```

Symlinks inside any NFS export must be **relative** or they break on clients.

## Companion model

`drowzeys/keys-latest-GLM-5.2-Quantrio-INT4-INT8-Mixed-Abliterated-DFlash` (also cached on bdea) is the GLM-5.2 equivalent — serve via the GLM-5.2 multi-node path (see the `glm-5.2-quanttrio-4x-dgx-spark` repo).
