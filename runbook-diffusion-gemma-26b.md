# Runbook: DiffusionGemma 26B-A4B on DGX Spark (single node, transformers)

**Model:** google/diffusiongemma-26B-A4B-it (26B MoE / 4B active, block-diffusion text generation, multimodal)
**Hardware:** 1× DGX Spark (GB10) — runs entirely on one node, no cluster needed
**Performance:** ~119 tok/s average (62–252 depending on task; diffusion adapts denoising steps to difficulty)
**Engine:** ⚠️ **transformers only** — vLLM and llama.cpp do NOT support this architecture
**Server:** custom OpenAI-compatible FastAPI server at `~/diffusion_gemma_server.py`
**Validated:** June 2026, edgexpert-9105

---

## Why this model

Diffusion-based text generation: instead of one token per forward pass, it denoises blocks of tokens. Simple prompts use few steps (→ very fast, 252 tok/s observed), complex ones use more. Google claims 1100+ tok/s on H100 FP8; GB10 hits ~11–23% of that. Still the fastest 26B-class model we've run on a single Spark for short answers.

## Engine support reality (June 2026)

| Engine | Status |
|---|---|
| llama.cpp | ❌ `diffusion-gemma` arch not in master (GGUFs exist on unsloth but nothing can run them) |
| vLLM 0.22.1 | ❌ crashes on MoE init (`TransformersMultiModalMoEForCausalLM` fallback fails) |
| **transformers 5.11+** | ✅ works via `DiffusionGemmaForBlockDiffusion` |

Recheck vLLM/llama.cpp support before assuming this is still true — arch support may have landed since.

## Step-by-Step

### 1. Create the venv (first time only)

```bash
uv venv diffusion-gemma --python 3.12
uv pip install -p ~/diffusion-gemma/bin/python \
  torch transformers accelerate sentencepiece protobuf pillow torchvision fastapi uvicorn
```

⚠️ `pillow` + `torchvision` are required by `Gemma4Processor` but NOT auto-installed — missing them gives a cryptic import error at load time.

### 2. Download the model (~71 GB)

```bash
hf download google/diffusiongemma-26B-A4B-it
```

### 3. Launch the server

```bash
nohup ~/diffusion-gemma/bin/python ~/diffusion_gemma_server.py --port 8000 > /tmp/diffusion-gemma.log 2>&1 &
```

VRAM: ~48 GB bfloat16 — fits comfortably on one GB10 alongside desktop/etc.

### 4. Verify

```bash
curl -s http://192.168.1.44:8000/v1/models
curl -s http://192.168.1.44:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/diffusiongemma-26B-A4B-it","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

OpenWebUI: base URL `http://192.168.1.44:8000/v1` (or Tailscale IP), model ID `google/diffusiongemma-26B-A4B-it`.

## Critical pitfalls (all learned the hard way)

1. **`device_map="auto"` breaks on GB10** — creates meta tensors → `RuntimeError: Tensor.item() cannot be called on meta tensors`. MUST use `device_map={"": "cuda:0"}` explicitly.
2. **`model.generate()` does NOT return a tensor** — it returns `DiffusionGemmaGenerationOutput`. Access `.sequences` for tokens and `.tokens_per_forward` (wrap in `float()`) for the diffusion-steps metric. Treating the output as a tensor crashes the server on first request.
3. **Port collision** — the default vLLM models also serve on 8000. Don't run DiffusionGemma simultaneously with another server on the same port (use `--port 8100` or similar if both must be up).

## Benchmark results (GB10, bfloat16, June 2026)

| Test | tok/s | Tokens/Forward |
|---|---|---|
| Short answer | 252 | 2.3 |
| Medium response | 62 | 19.7 |
| Creative writing | 88 | 7.6 |
| Reasoning | 91 | 26.6 |
| Code generation | 102 | 36.6 |
| **Average** | **119** | — |

`tokens_per_forward` = effective parallelism of the diffusion decode — the key metric to watch when tuning.

## Files

- Server: `~/diffusion_gemma_server.py` (endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- Venv: `~/diffusion-gemma/`
- Reference notes: `references/diffusion-gemma.md` (dgx-spark skill)
