# End-to-End Model Serving Recipe Playbook
## From community research to a verified, reproducible production deployment

**Basis:** the MiaAI-Lab + tonyd2wild strategy v2 developed in this conversation, including repository updates reviewed from September 2026.

**Deliverable status:** operational playbook, not an executed deployment. Commands below are read-only examples or explicitly marked templates. Recipe scripts, benchmark runners and deployment profiles described here are requirements to implement for each target stack—not files claimed to exist already. No servers were restarted or reconfigured to produce this document.

**Objective:** maximize correctly completed application tasks at the required latency, context and concurrency, within approved hardware and operational constraints. Peak synthetic tokens/second is diagnostic, not the release criterion.

---

## 1. Operating contract

Every candidate passes the same sequence:

**Scope → source audit → baseline capture → hardware/fabric proof → artifact validation → baseline serving → correctness → optimization → load/soak → packaging → canary → maintenance.**

A failed gate stops promotion. It does not justify borrowing another node, disabling host safeguards, quietly dropping vision, changing the checkpoint or reducing required context.

### Non-negotiable rules

1. Only use explicitly approved nodes and GPUs. Discover current ownership before any workload runs.
2. Preserve the known-good deployment and its effective launch configuration before changing it.
3. Distinguish author-reported, locally reproduced, estimated, untested and refuted findings.
4. Pin the complete serving configuration, not just the model name.
5. Inspect actual kernels, tensor placement and API behavior; flags can be ignored or route to fallbacks.
6. Change one variable at a time for screening, then test important interactions.
7. Keep cold-cache and warm-cache experiments separate.
8. Never substitute boot success for depth, concurrency or correctness validation.
9. Roll back on regression; do not repair a failing experiment by modifying the reference baseline.
10. Use scoped process/container cleanup. No global orphan sweeps, cache eviction, OOM-protection changes or shared-service restarts by default.

**Fleet-specific boundary:** retain the dedicated MiniMax H3 placement; do not deploy H3 on bdea. Co-tenancy in someone else's recipe is evidence for an optional future experiment, not permission to change this boundary.

---

## 2. Required outputs

At completion, a recipe must include:

```text
recipe/
├── README.md
├── acceptance.yaml                 # agreed requirements and promotion gates
├── recipe.lock.json                # immutable artifact/runtime identity
├── .env.example                    # safe placeholders; no credentials
├── profiles/
│   ├── interactive.env
│   ├── long-context.env
│   ├── throughput.env
│   └── reference.env
├── doctor.sh                       # non-destructive preflight
├── start.sh                        # scoped, idempotent, logs effective argv
├── stop.sh                         # only owned resources
├── rollback.sh                     # previous pinned release
├── patches/                        # compatibility guards + upstream attribution
├── tests/                          # API, tools, quality, depth, mixed load
├── research/
│   ├── source-ledger.md
│   ├── experiment-plan.md
│   └── negative-results.md
└── results/<run-id>/
    ├── manifest.json
    ├── effective-launch.txt
    ├── preflight.json
    ├── requests.jsonl
    ├── responses.jsonl
    ├── metrics.json
    ├── server.log
    └── verdict.md
```

This is the target package contract, not an assertion that these files already exist. Private request/response artifacts need restricted access and redaction before publication.

---

## 3. Phase 0 — Define scope and success

### Inputs
- Application: interactive coding, tools, general chat, long-document analysis, multimodal, batch work.
- Existing serving configuration, if any.
- Approved machines, GPU IDs, maintenance window and services that must remain untouched.
- Real request traces where available, sanitized before analysis.

### Actions
1. Write the resource allowlist and service exclusions.
2. Define native model context separately from any proposed extrapolated context.
3. Record representative prompt lengths, requested output lengths, active conversations and arrival patterns.
4. Define required API features: streaming, reasoning, thinking controls, tool choice, typed arguments, images/video, logprobs if used.
5. Set acceptance thresholds before testing. Leave unknowns explicitly unresolved rather than inventing SLOs.
6. Set storage, memory-headroom, thermal, error-rate and rollback requirements.

### Acceptance template

```yaml
# TEMPLATE: replace every REQUIRED value before a deployment is approved.
resource_scope:
  approved_nodes: REQUIRED
  approved_gpu_ids: REQUIRED
  excluded_services: REQUIRED
  maintenance_window: REQUIRED
workload:
  prompt_length_distribution: REQUIRED
  output_budget_distribution: REQUIRED
  required_context_tokens: REQUIRED
  target_concurrency: REQUIRED
  arrival_pattern: REQUIRED
  modalities: REQUIRED
  tools_required: true
slo:
  warm_ttft_p95_seconds: REQUIRED
  cold_ttft_p95_seconds_by_depth: REQUIRED
  task_completion_p95_seconds: REQUIRED
  minimum_quality_score_and_suite: REQUIRED
  maximum_error_rate: REQUIRED
  minimum_memory_headroom_per_rank: REQUIRED
  soak_duration_and_request_count: REQUIRED
release:
  rollback_trigger: REQUIRED
  rollback_time_target_seconds: REQUIRED
  owner: REQUIRED
```

**Gate 0:** requirements and resource boundaries are explicit. Deployment work is blocked while essential approvals or thresholds remain unresolved.

---

## 4. Phase 1 — Research and extract transferable findings

### Read in this order
1. Repository README: current default, dated updates, historical configurations and limitations.
2. Launcher: actual defaults, precedence, model paths, flags and process ownership.
3. Checkpoint config/index: architecture, precision and artifact composition.
4. Dockerfile/runtime pins and patch files.
5. Raw benchmark receipts and test harness.
6. Issues/commits covering known failures and retired workarounds.

A repository's updated timestamp is a discovery hint, not proof that its serving recipe changed. The research underlying this playbook used repository listings and README excerpts; source/patch/receipt verification remains a required step for implementation.

### Source ledger fields

```text
source URL + commit/revision
retrieved timestamp
claim
hardware/topology
checkpoint + quantization implementation
runtime + kernel + patch identity
sampler/thinking/prompt/concurrency/cache conditions
raw receipt available? inspected?
evidence: author-reported | locally reproduced | estimated | untested | refuted
transfer condition
local CONFIRM test
local REFUTE test
credit/license
```

### Questions that prevent bad transfers
- Same model architecture, or merely a similar name?
- Same GPU compute capability, driver/toolchain and attention implementation?
- Same checkpoint conversion and activation precision?
- Same transport and physical topology?
- Same graph mode, draft head and recurrent-state dtype?
- Same prompt class and client-side token accounting?
- Is the advertised feature implemented, or only an accepted inert knob?

**Gate 1:** every proposed optimization has a mechanism, compatibility scope and falsifiable local test. Do not transplant full model files between unrelated engine revisions.

---

## 5. Phase 2 — Capture the current working deployment

Do this before downloading dependencies into its environment, replacing overlays or stopping it.

### Capture
- Process argv, service manager and container/Compose ownership.
- Image digest, engine commit, dependency versions and mounted patches.
- Checkpoint revision, tokenizer, template and generation config.
- GPU/node mapping and fabric environment.
- Endpoint model ID, parser settings and required authentication.
- Model initialization log, cache allocation and real inference response.
- Current memory/headroom, representative quality and latency baseline.
- Exact restart and rollback procedure.

### Read-only discovery examples

```bash
uname -m
nvidia-smi
free -h
ss -ltnp
ps -eo pid,ppid,args
# If Docker is the deployment mechanism:
docker ps --no-trunc
docker inspect "$CONTAINER_NAME"
```

Inspect output may contain credentials or private paths. Store restricted originals and redact public receipts. Do not assume a directory containing a Compose file is the project that owns the live container; inspect its labels.

### Rollback proof
A rollback package needs the old immutable image/config/patches and accessible model revision. An untested remembered launch command is not a rollback plan. Exercise restart/recovery in an approved window or isolated environment.

**Gate 2:** current deployment is identifiable and recoverable. No experimentation in its environment before this gate.

---

## 6. Phase 3 — Hardware, memory and fabric preflight

### Hardware per approved node

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version --format=csv
nvidia-smi topo -m
free -h
lscpu
ip -br addr
ip route
# When RDMA tools are installed:
rdma link
ibv_devinfo
```

Unavailable telemetry must be marked unavailable, not converted to zero. On unified-memory machines, device and host counters can overlap or have different accounting semantics.

### Budget memory by component

```text
resident weights
+ target KV / sparse-attention state
+ recurrent or Mamba/DeltaNet state
+ draft weights / draft KV / draft recurrent state
+ activation and temporary workspaces
+ CUDA graph pools
+ communication buffers
+ vision/audio/video components
+ API/runtime/OS reserve
+ explicit operational headroom
```

Also measure the transient peaks of download conversion, CPU repacking, model load and graph capture. A model may fit after loading and still fail during load.

For UMA, CPU offload is placement within shared physical capacity, not extra RAM. Reclaimable file cache and GPU allocations must not be counted as independent free pools. Capacity is constrained by the most burdened rank, not fleet-total free memory.

### Prove the transport before model loading
1. Resolve each rank to its real interface, HCA, GID and route.
2. Verify MTU and link state on the intended fabric.
3. Run pairwise RDMA tests where applicable and NCCL collectives across the exact approved topology.
4. Measure small-message latency as well as large-message bandwidth; decode collectives and checkpoint transfer are different workloads.
5. Capture collective-library identity and logs proving the selected transport.
6. Do not alter switch routes, firewall policy or global NCCL libraries without a separately scoped change.

Use available high-speed links for transfers after verifying them. A nominal 200G link does not establish achieved throughput. Run network stress checks in an approved window if other services share the fabric.

### Local weights versus shared weights
- Local copy: storage/transfer cost; less dependency on a remote file server.
- Read-only NFS: fewer copies; network and server availability affect loading and potentially serving.
- mmap lookup tables: may remain dependent on storage throughout serving, not just startup.

Verify file sizes and hashes against trusted artifact metadata. Prefer resumed downloads, atomic staging and verified revisions; file existence is not integrity proof.

**Gate 3:** every rank has measured capacity, usable storage and validated transport. Do not load a distributed model to discover a basic fabric failure.

---

## 7. Phase 4 — Select the checkpoint/runtime/kernel combination

### Architecture audit
Inventory:
- Routed versus shared experts; routing top-k and any padding.
- Dense attention versus sparse/MLA layers.
- Recurrent layers and state precision.
- Embeddings, output heads, lookup-only PLE/n-gram tables.
- MTP or external drafter and tensor sharing.
- Vision/audio components and preprocessing limits.
- Cache grouping, block geometry and supported context extension.

### Selection rules
1. Prefer a known-compatible, minimally patched configuration—not a preferred quantization label in isolation.
2. Inspect actual W/A precision, group sizes, scale formats and quantization ignore lists.
3. Confirm the executed MoE/attention/cache kernels on the target SM architecture.
4. Evaluate quality against the highest practical trusted reference, noting when a reference itself is quantized or cross-stack.
5. Treat a fine-tuned or modified checkpoint as a new artifact, including drafter compatibility, not a guaranteed drop-in swap.
6. Separate stored KV format from compute precision. Software-packed FP4 cache is not synonymous with native FP4 matrix multiplication.
7. For large sparse lookup tables, investigate memory-mapped/staged access and compiler duplication before rejecting the fit. Do not generalize that mechanism to dense-layer offload.

### Candidate profiles
- **Interactive:** low warm TTFT, correct tools, stable long-history turns.
- **Long-context:** verified depth and enough resident capacity for required active conversations.
- **Throughput:** compare wider distributed execution with independent replicas and conversation affinity.
- **Reference:** simpler or higher-fidelity configuration for diagnosing regressions.

Do not assume a standalone EXL3 endpoint batches requests because it accepts concurrent HTTP calls. Do not assume TP4 is faster than TP2, or that a Spark RoCE result transfers to NVLink-paired 3090s.

**Gate 4:** candidate fits both artifact compatibility and the application contract. Unsupported features are disqualifying unless requirements are explicitly changed.

---

## 8. Phase 5 — Build, launch and establish baseline readiness

### Build requirements
- Pin source commits and base image digests.
- Record compiler, CUDA, PyTorch and kernel-library versions.
- Build for the correct architecture and validate runtime driver compatibility.
- Guard patches by expected source hashes or exact revision compatibility.
- Fail closed on patch mismatches; do not silently fuzzy-patch a serving kernel.
- Verify the same required build artifacts on all ranks.

### Launcher behavior
1. Run non-destructive preflight.
2. Reject occupied ports or conflicting GPU ownership.
3. Validate checkpoint completeness.
4. Render and retain the effective arguments and environment with secrets redacted.
5. Start only recipe-owned resources, using the chosen runtime's documented rank ordering.
6. Wait for bounded readiness signals with observable logs.
7. Exit with a meaningful status and recovery instructions on timeout.

Idempotence means an existing matching healthy deployment is recognized. An unrelated process is not silently killed, and a mismatched existing deployment is not silently accepted.

### Readiness levels
Report each separately:
1. Process alive.
2. Port listening.
3. Model initialized.
4. First inference succeeded.
5. Required API behavior passed.
6. Required context/concurrency passed.
7. Any fallback, offloaded state or degraded modality.

### Minimal HTTP examples

```bash
# TEMPLATE: exported variables must identify the intended authenticated endpoint.
# API_BASE ends in /v1. AUTH_HEADER_FILE points to a restricted, temporary
# header file provisioned securely with the endpoint's authorization header.
# Never commit or publish that file; remove it securely according to local policy.
curl --fail-with-body --max-time 30 \
  --header @"${AUTH_HEADER_FILE}" \
  "${API_BASE}/models"

# smoke-request.json is authored for this endpoint's real model ID and budget.
curl --fail-with-body --max-time 180 \
  --header @"${AUTH_HEADER_FILE}" \
  -H 'Content-Type: application/json' \
  --data-binary @smoke-request.json \
  "${API_BASE}/chat/completions"
```

Use a sufficiently large output budget for reasoning models. Empty visible content can mean reasoning consumed the budget, not that inference failed. Verify GPU residency/activity and draft acceptance counters: a working API can silently run on CPU or with speculation disabled.

**Gate 5:** the baseline generates a correct response using the intended hardware and execution path. No optimization results count before this gate.

---

## 9. Phase 6 — Correctness and API contract suite

### A. Basic response semantics
- Non-streaming and streaming answers are well-formed.
- Model ID, finish reasons, usage and error responses are handled by the actual client.
- Reasoning fields are correctly separated and replayed where required.
- Thinking controls are tested empirically; defaults and hard restrictions are distinguished.
- Token-budget exhaustion is recorded separately from substantive answer correctness.

### B. Tool workflows
- Required and optional arguments, enums, numbers, booleans and nested objects.
- Automatic tool selection and required/forced selection where promised.
- Tool calls split across streamed deltas.
- Tool-result replay followed by the final answer.
- Multiple tool calls when supported and multi-turn histories with reasoning.
- Real application system prompt and schemas, not only toy prompts.

### C. Output integrity and quality
- Multilingual samples, including scripts that exposed corruption in community reports.
- Unexpected replacement characters, malformed tool payloads and repetition loops.
- Reasoning, coding and structured tasks with objective checks where possible.
- Compare reference and candidate on the same item set and sampling policy.
- Inspect suspicious scores manually; scorer normalization or a broken harness can manufacture apparent model failures.

### D. Multimodal behavior
- Image/video types and sizes at documented limits.
- Text + image + tools in the same supported request path.
- Bounded errors for unsupported inputs.
- Peak memory during processing, not just a text-only initialization budget.

A clean multilingual probe is not proof of universal corruption freedom. A successful generation comparison is not proof that a quantized cache is universally lossless.

**Gate 6:** required client semantics and agreed quality thresholds pass. A faster configuration with broken tools does not advance.

---

## 10. Phase 7 — Validate context and concurrency honestly

### Four separate fields in every report

```text
native_context_limit
configured_context_limit
allocated_cache_pool_tokens_and_geometry
deepest_validated_prompt_tokens + generated_output_tokens
```

Also record scheduler sequence limits and empirically supported concurrent session lengths. Pool divided by context is at most a rough capacity estimate: draft state, hybrid geometry, fragmentation, prefix reuse and other limits can invalidate it.

### Context ladder
1. Start with short prompts that already pass correctness.
2. Increase to representative production depths.
3. Approach the required maximum while reserving output tokens inside the context window.
4. Count the rendered prompt with the actual tokenizer/template or trusted server usage—not character count.
5. Use unique randomized prefixes for cold tests and verify cache-miss behavior where observable.
6. Plant retrieval targets at early, middle and late positions with distractors.
7. Add multiple targets, cross-document questions and long-context tool workflows.
8. Repeat warm-prefix turns after deep prefills to catch workspace-lock and reuse bugs.
9. Run mixed short/deep concurrent requests and cancellation/recovery.

Extrapolated context via YaRN requires its own quality tests. Allocating a million-token window does not establish useful million-token behavior.

### Cache conditions
- **Cold:** unique prefix, known model/kernel warmup state; document any residual cache ambiguity.
- **Warm:** intentionally shared prefix, measured hit/reuse behavior.
- **Restart-cold:** includes compilation/loading/page-cache effects and is reported separately.

Do not clear the host's global page cache merely to make a benchmark cold. Use request-level isolation, and schedule any truly necessary system-level cold-start experiment separately.

**Gate 7:** actual requests at required depth and concurrency pass, with memory and service health intact afterward.

---

## 11. Phase 8 — Optimize using a controlled experiment ladder

### Experiment card

```text
ID / hypothesis / source attribution
reference configuration ID
single primary change
predicted mechanism
CONFIRM and REFUTE criteria
quality and safety guards
workload and sampler
cold/warm state
repetitions and ordering
resource allowlist
rollback configuration
result + limitations + raw receipt paths
```

### Recommended order

**1. Kernel selection and placement**
- Confirm intended quantized expert and attention kernels.
- Investigate compiler-induced copies and unnecessary PLE/embedding materialization.
- Compare TP-only versus TP+EP only where supported and justified.

**2. Memory/state precision**
- Evaluate checkpoint alternatives with actual quality and load-memory tests.
- Test recurrent-state dtype separately from target KV dtype.
- Keep attention/heads at higher precision when beneficial, not as an unexamined rule.

**3. Cache allocation and geometry**
- Compare supported target KV formats and draft-cache costs.
- Measure pool changes after graph capture and under load.
- Test explicit byte budgets only when their semantics in that engine revision are known.
- Retain headroom for the busiest rank and input modality.

**4. Prefill and scheduler**
- Sweep supported chunk sizes and sequence caps.
- Measure deep-prefill latency, decode starvation, queueing and memory together.
- Smaller chunks may free KV; larger chunks may improve prefill. Neither is universally correct.

**5. CUDA graphs**
- Compare eager, piecewise and full-plus-piecewise where supported.
- Confirm graph capture actually covers decode/draft shapes.
- Test both cold prefills and repeated warm turns.
- A b12x-specific eager requirement does not automatically apply to a Marlin lane.

**6. Speculative decoding**
- Always include speculation off.
- Compare supported MTP, DFlash2, DSpark and lightweight n-gram candidates.
- Record accepted tokens per target pass and the exact counter definition.
- Test realistic prose, code, tools and structured output at relevant sampling and concurrency.
- Account for draft weights, draft KV, verifier workspace and any reduction in useful cache capacity.
- If engine, quant and drafter all change, label it a whole-configuration comparison, not an isolated drafter result.

**7. Parallelism and replicas**
- Compare wider TP/EP/DCP to smaller groups/replicas on the real fabric.
- Verify each runtime's compatibility before introducing pipeline or decode-context parallelism.
- Use conversation affinity for prefix warmth when routing replicas.
- Test failover semantics; do not automatically retry side-effecting application tool actions blindly.

### Interaction tests after screening
- Speculation × graph mode × concurrency.
- Prefill chunk × KV budget × warm-prefix reuse.
- Recurrent/KV precision × context depth × quality.
- Parallelism × fabric topology × request length.

### Stop conditions
OOM, output corruption, wrong kernel/fallback, tool regression, exceeded headroom, unacceptable tail latency, broken modality or unexplained nondeterministic failures stop promotion immediately. Preserve logs and return to the pinned reference; do not pile on additional changes.

**Gate 8:** winning profiles improve the agreed workload objective and pass all earlier gates again.

---

## 12. Phase 9 — Benchmark accounting, load and soak

### Minimum workload dimensions
- Prose/chat, code, tool round-trips and structured output.
- Counting/repetition only as a separately labeled synthetic ceiling.
- Short, representative and required-maximum prompt depths.
- Cold and warm prefix conditions.
- Single-stream and workload-relevant concurrency.
- Mixed-length arrivals, not only synchronized identical requests.
- Thinking settings and samplers used by the real application.

### Metric definitions
- **TTFT:** request submission to first declared token event; distinguish any generated token from first visible-answer token.
- **End-to-end latency:** submission to completed response, including queueing.
- **Decode throughput:** define which tokens and time interval are used. With speculation, streamed chunks may contain multiple tokens.
- **Aggregate throughput:** total eligible completion tokens over a shared wall-clock measurement interval; do not sum incompatible per-request rates.
- **Prefill proxy:** prompt tokens divided by TTFT includes queueing and other overhead; label it as a proxy unless engine timing isolates prefill.
- **Task throughput:** correctly completed tasks per wall-clock interval.
- **Accepted draft tokens:** report exact engine metric semantics, not an unlabeled acceptance percentage.

Use server/tokenizer token counts rather than streamed character counts. Separate reasoning and visible tokens when measurable. Count errors, truncations, cancellations and retries; excluding them can inflate a benchmark.

### Experimental hygiene
1. Keep the reference intact and use identical evaluation items.
2. Predeclare the primary objective and thresholds.
3. Warm kernels in a documented manner.
4. Interleave/randomize A/B ordering to reduce thermal/cache/time drift.
5. Repeat runs; publish sample counts and variation.
6. Do not report a trustworthy p99 from a handful of requests.
7. For quality, prefer paired item-level analysis and inspect disagreements.
8. Keep whole-configuration results separate from causal single-variable claims.

### Soak and recovery
Run the duration/request volume agreed in Phase 0 with mixed prompt lengths, warm histories and required modalities. Monitor memory growth, thermals, queue size, preemptions, errors, TTFT and task success.

In an approved test environment/window, exercise client disconnect, cancellation, worker failure, restart and rollback. Verify that stop scripts do not affect other services. Co-tenancy requires a separate approval and mixed-load SLO test, not extrapolation from a short counting benchmark.

**Gate 9:** sustained performance and recovery meet requirements without silent degradation.

---

## 13. Phase 10 — Package, canary and promote

### Release manifest
Record:
- Hardware/topology and resource scope.
- Checkpoint ID/revision and file integrity manifest.
- Tokenizer/template/parser versions and generation defaults.
- Image digest, source commits, CUDA/driver compatibility and build recipe.
- Patch hashes, affected upstream files, guards and credit.
- Exact effective launch command per rank.
- Cache/state/graph/speculation configuration.
- Per-rank load and serving peaks, operational headroom.
- Validated context, concurrency, modalities and API features.
- Benchmark receipts and known limitations.
- Previous release ID and rollback instructions.

### Promotion procedure
1. Freeze the candidate configuration; no late unmeasured edits.
2. Validate from a clean launch using the packaged scripts.
3. Re-run representative correctness and inference checks.
4. Canary a bounded slice of approved traffic with reference comparison.
5. Observe SLOs and error/quality signals over the agreed window.
6. Promote only after all gates pass.
7. Keep previous artifacts and model revisions available until rollback retention expires.

### Rollback triggers
Use predeclared thresholds for correctness, error rate, latency, memory pressure and availability. An unexpected CPU fallback, silent draft loss, token corruption or tool-parser failure warrants immediate investigation even if HTTP health remains green.

### Public endpoint safety
Keep services on approved private interfaces by default. If external access is needed, use authenticated HTTPS and explicit network restrictions. Do not expose distributed control planes, file shares or privileged dashboards to the public internet as part of serving an API.

**Gate 10:** a clean launch, canary and rollback are verified. Only then label the profile production-ready.

---

## 14. Phase 11 — Maintain the recipe and retire workarounds

### On each relevant upstream update
1. Fetch source/launcher/patch changes and identify actual behavioral deltas.
2. Check whether old workarounds are fixed upstream, obsolete or incompatible.
3. Rebuild an isolated candidate with fresh immutable pins.
4. Re-run affected regression tests and the common acceptance suite.
5. Compare to the existing baseline under the same conditions.
6. Promote only after canary; otherwise record the negative result and retain the old release.

Do not automatically deploy a new community default. The current README, a historical receipt and a staged estimate can describe three different configurations.

### Monitoring minimum
- Process/model readiness and GPU activity.
- Queueing, TTFT, inter-token and E2E latency.
- Task/tool success and malformed response rates.
- Cache utilization/reuse, preemptions and draft acceptance.
- Per-node available memory, sustained growth and thermals.
- Fabric/storage errors and external dependency availability.

### Negative-results record
Record failed experiments with exact scope, symptoms, reproduction conditions and rollback. Mark untestable claims and contradicted hypotheses explicitly. A diagnostic tool must declare which checks it implements; green coverage of a subset is not full certification.

---

## 15. Applying the playbook to our existing recipes

### First pass — correctness and provenance
- Preserve known-good configurations and capture actual launch ownership.
- Replace blanket recommendations with backend/version-specific conditions.
- Add common tool, streaming, multilingual and long-history checks.
- Separate advertised context from locally validated depth.

### Second pass — memory and kernel efficiency
- Audit target/draft memory, recurrent state, graph reserves and load-time peaks.
- For Flash-Next candidates, prioritize lookup-table placement, compiler duplication and intended expert kernels.
- For GLM candidates, test checkpoint-specific corruption and graph/prefill interactions.
- Never copy GLM Flash patches into full GLM solely because names or filenames resemble one another.

### Third pass — fleet optimization
- Compare replicas versus wider distributed execution on approved hardware only.
- Favor the profile that improves real agent turns at the required history depth.
- Preserve dedicated H3 isolation; do not reclaim its node implicitly.
- Retain a simple reference profile alongside aggressively optimized profiles.

---

## 16. Release checklist

- [ ] Approved nodes, GPUs, exclusions and maintenance scope recorded.
- [ ] Workload, SLOs, quality threshold and soak requirements agreed.
- [ ] Source claims attributed and compatibility assumptions tested.
- [ ] Previous deployment captured and rollback exercised.
- [ ] Every rank's memory/load peaks, storage and transport validated.
- [ ] Checkpoint/runtime/tokenizer/template/patch identities pinned.
- [ ] Actual GPU and kernel paths confirmed; no silent fallback.
- [ ] Streaming, tools, reasoning controls and multilingual probes pass.
- [ ] Required modalities pass at their declared limits.
- [ ] Required depth validated with real token counts and output reserve.
- [ ] Cold/warm and concurrent mixed-length traffic tested.
- [ ] Speculation and graph choices backed by workload-specific receipts.
- [ ] Throughput denominators and sample sizes documented.
- [ ] Soak, cancellation, recovery and rollback pass.
- [ ] Clean packaged launch and bounded canary pass.
- [ ] Public docs redact secrets and sensitive requests.
- [ ] Current default, alternatives, historical results and estimates clearly separated.

**Definition of done:** someone else can launch the pinned profile on matching approved hardware, reproduce its declared acceptance tests, understand its limitations and restore the previous working profile without reverse-engineering our session.

---

## 17. Sources and attribution

The strategy combines Mia's layered optimization/reproducibility patterns with Tony's workload-specific comparisons, memory ladders and explicit failure/evidence taxonomy. Individual checkpoints, kernels and transport implementations retain their original authorship; repository ownership alone does not establish authorship of every component.

### MiaAI-Lab
- [Qwen3.8 Flash-Next dual Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks): memory variants, rendered launch configuration, preflight, optional NFS.
- [GLM-5.3 Flash EXL3 dual Spark](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks): expert integration, cache geometry, evolving configuration receipts.
- [GLM-5.3 Flash switchless deployment](https://github.com/MiaAI-Lab/glm-5.3-flash-4x-dgx-spark-switchless): patched NCCL transport and real agent workload; repository credits Alex Ellis/OpenFaaS for production operation.
- [Qwen3.8-27B EXL3 deployment](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw): memory/speculation tradeoffs and explicit batch-one/API limitations.
- [ExLlamaV3 fork](https://github.com/MiaAI-Lab/exllamav3): runtime-specific cache and drafting implementation.
- [sparkDash](https://github.com/MiaAI-Lab/sparkDash): cold/warm performance observability and operational metrics.

### tonyd2wild
- [Qwen38 Flash-Next 4×3090](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090): staged lookup table, expert-kernel routing, prompt-dependent speed and fallback lanes.
- [Qwen3.8-27B 3090 cookbook](https://github.com/tonyd2wild/Qwen3.8-27B-3090-Cookbook): topology/profile selection and boot-by-boot memory ladder.
- [SGLang versus vLLM](https://github.com/tonyd2wild/Qwen3.8-27B-SGLang-vs-vLLM-2x3090): workload-dependent whole-configuration comparisons and capacity tradeoffs.
- [GLM-5.3 Flash dual Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark): checkpoint-specific output corruption reports and regression motivation.
- [GLM-5.3 Flash TP4](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark): scheduler, graph, prefill and memory-budget revisions.
- [Model serving minefield](https://github.com/tonyd2wild/model-serving-minefield): evidence labels, silent failures and declared diagnostic coverage.
- [DS4/H3 co-tenancy](https://github.com/tonyd2wild/DS4-H3-Video-Gen-Factory): workload-conditioned contention measurements, not blanket co-tenancy approval.

### Local provenance
- Original reverse-engineered strategy: `mia-recipe-strategy.md`.
- Updated synthesis: `mia-tony-recipe-strategy-v2.md`.

External findings remain author-reported until reproduced under this playbook. This document does not certify any author's latest configuration or claim fresh performance results on our fleet.
