# Runbook — GLM-5.3-Flash-DERISKED-NVFP4 on 4× DGX Spark (TP4, NFSoRDMA)

Full campaign repo: https://github.com/chishiki37/glm-5.3-flash-derisked-nvfp4-4x-dgx-spark

## Standing config (the only working one)

```
Image: radixark/vllm-glm53-flash:sm121-v8 (locally built v1→v8 chain; hub repo is PRIVATE)
GLM_SPEC=0                      # MTP unbootable — draft head unquantized, see below
GLM_KV_MEM=9663676416           # 9 GiB fp8 pin
GLM_KV_DTYPE=fp8_e4m3
GLM_MOE_BACKEND=marlin          # default
TP4: head 10.10.10.14 + workers 10.10.10.12 / 10.10.10.1 / 10.10.10.2
Weights: NFSoRDMA from ae1e (10.10.10.13), mount /var/tmp/models-ae1e (ro,proto=rdma,hard,vers=3)
```

Measured: C1 18.95 tok/s · C4 agg 66.63 · C8 agg 79.43 (256-tok battery, temp 0, thinking off).
Boot ~7 min to healthy with the flusher stack below.

## Bring-up prerequisites (hard-won, do not skip)

1. **Gentle flusher on every rank** (`gentle_flusher.sh` in the campaign repo):
   drop_caches ONLY when MemAvailable < 8 GiB, 30s check. Periodic 12s flushing
   livelocks the mmap weight load; no flushing starves NVRM (GB10 allocates from
   MemFree, fails fast) → head-node lockout, physical power cycle.
2. `vm.vfs_cache_pressure=200` on all nodes (persist via sysctl).
3. Sudoers drop-in per node for the flusher:
   `vikassridhar ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches, /sbin/sysctl -w vm.vfs_cache_pressure=200`
4. **No GPU co-tenants.** A leftover serve on any rank kills TP4 boot
   (NVML "Unknown Error" / WorkerProc init OOM). Check `docker ps` on all 4 first.
5. Pre-launch guard: MemAvailable ≥ util×MemTotal + 6 GiB on ALL nodes.
6. cb98's /tmp is tmpfs — re-scp the launcher after any reboot.

## MTP dead-end (8 arms, all DNB)

Checkpoint ships MTP draft head UNQUANTIZED while main MoE is NVFP4. One global
`--moe-backend`; no backend accepts both:
- marlin: rejects unquantized MoE (draft)
- auto→flashinfer_cutlass: JIT nvcc `fused_moe_120/deepgemm_jit_setup.cu` fails for sm_121a
- triton: rejects NvFP4 MoE (main)

Fix is weights-side: quantize the MTP head. Until then, DERISKED runs AR-only.
If MTP speed matters more than the derisked weights, the LibertAIDAI-NVFP4
TP4+MTP3 config (C1 40.0 / C8 114.4) is the fleet's GLM-5.3-Flash speed king.
