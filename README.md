# LLM VRAM / RAM / Token/s Calculator — v3.1 (fixed)

A browser-based, single-file capacity-planning tool for local LLM inference. The interface is in Hungarian; this README explains the calculation model and the v3.1 fixes in English.

## Run

1. Open `index.html` (or `llm-vram-calculator_v3_1_fixed.html`) in a modern browser. There is no backend or build step.
2. Choose a model, or import its Hugging Face `config.json` / paste configuration JSON.
3. Set weight quantization, KV-cache type, context length, concurrent requests, and memory safety margin.
4. Choose hardware (or edit usable memory, bandwidth, compute, RAM and RAM bandwidth).
5. Inspect weights, KV cache, total requirement, **fit / partial offload / no fit**, decode and prefill estimates, and the hardware comparison table.
6. To evaluate local ownership versus cloud API, enter daily tokens, cloud prices, hardware cost, power, and economic assumptions. **For discrete GPUs, complete the separate host-machine cost and power fields** to enable meaningful TCO/payback estimates.
7. Optionally calibrate throughput with `llama-bench` results. Export/import model, hardware and price data as JSON. A shareable fragment (`#...`) stores the scenario without local calibration data.

The page does not require a cloud inference API key. Hugging Face model lookup requires internet access from the browser. Otherwise it can be used offline, with fallback system fonts.

## Calculation overview

All memory calculations convert bytes into binary **GiB** (`2^30` bytes). Hardware bandwidth is expressed in decimal **GB/s** (`10^9` bytes/s).

### Weights and RAM placement

```text
host_weight_bytes      = host_parameters_B * 1e9 * effective_bits_per_weight / 8
accelerator_weight_B   = (total_parameters_B - host_parameters_B) * 1e9
                         * effective_bits_per_weight / 8
                         + speculative_draft_memory
```

The `host` parameter count is a *subset* of total parameters, not an additional set. v3.1 prevents double-counting it for hybrid architectures that store some parameters in system RAM.

`runtime_overhead` is a coarse estimate containing backend/device overhead, a logits buffer of `512 * vocab_size * 4` bytes, and an activation allowance. For multi-GPU configurations, the usable capacity reserves an extra **0.5 GiB per GPU** in addition to the modeled runtime estimate.

### KV cache

For one sequence (bytes, not GiB):

```text
full-attention KV = full_layers * 2 * kv_heads * head_dim * context * bytes_per_KV_element
SWA KV            = SWA_layers  * 2 * kv_heads * head_dim * min(context, window) * bytes_per_KV_element
MLA KV            = MLA_layers  * latent_dim * context * bytes_per_KV_element
linear/recurrent  = fixed_state_bytes (where known)
```

The per-sequence cache is multiplied by the number of simultaneous requests. For full attention with a sparse read cap, the cap limits *read traffic*, not necessarily stored KV-memory capacity.

### Fit and maximum context

The memory requirement is the estimated weight + cache + runtime footprint with the configured safety margin. Discrete GPUs can be classified as **partial offload** if the excess weight bytes fit into system RAM after reserving 8 GiB for the host OS and accounting for any model weights already resident there.

**v3.1 computes maximum context using binary search against the same `placement()` decision used for the fit status.** It reports two limits:

- **Without offload:** all accelerator-resident model weights must fit in GPU/accelerator memory.
- **With CPU offload:** for discrete GPU configurations only, a larger context may fit with some weights in system RAM.

This fixes pure sliding-window models and hybrid full-attention/SWA models whose cache growth changes when the sliding window saturates.

### Throughput and TTFT

Decode throughput remains a **heuristic** memory-bandwidth/compute model using active parameter bytes (important for MoE), KV bytes read during decode, hardware bandwidth efficiencies, and a per-layer fixed-cost approximation. For supported model configurations, speculative decoding applies a user-adjustable average accepted-tokens-per-step versus draft overhead estimate. Parallel throughput approximates weight reuse and additional KV/cache traffic.

Prompt/prefill processing uses an approximate compute model:

```text
prefill_tokens_per_second = effective_dense_FP16_FLOPS * 0.35 /
                            (2 * active_parameters)
TTFT_estimate             = prompt_tokens / prefill_tokens_per_second
                            + first_decode_token_seconds
```

**The first-token decode term is new in v3.1** and is intentionally not accelerated using the multi-token speculative decoding factor. Network latency, queueing, model loading and runtime startup are not modeled; the displayed TTFT is not an end-to-end serving guarantee.

KV quantization saves memory, but actual performance also depends on dequantization and attention kernel support. Memory savings should **not** be read as a guaranteed speedup. Runtime and GPU/kernel support depend on the selected backend; warnings are shown for potentially incompatible quantization/hardware combinations.

### Calibration

You can provide measured `pp512` and `tg128` rates from `llama-bench`. v3.1 scopes calibration to a **hardware + model + parameter count + weight quantization + KV type + multi-GPU configuration**, so a calibration for one model or quantization is no longer silently applied to all models on the same machine. Old hardware-only calibration entries are not automatically reused. Calibration is saved locally in the browser rather than in shared URLs.

### Cost and ROI

Cloud API monthly cost uses 30 days, daily input/output volumes, input/output list prices, and discounted cached input when configured. Local capacity is estimated from aggregate decode speed and uncached prefill time. The calculator allocates additional machines when estimated workload exceeds **85% of a 24-hour day** per machine.

```text
system_capex      = number_of_machines * (base_hardware_price + discrete_GPU_host_cost)
monthly_power     = 30 * machines * (busy_hours_per_machine * load_W
                    + idle_hours_per_machine * idle_W) / 1000 * electricity_price_per_kWh
monthly_TCO       = system_capex * (1 - residual_value_ratio) / amortization_months
                    + monthly_power + monthly_operations
payback_months    = system_capex / (monthly_cloud_cost - monthly_power - monthly_operations)
```

A number of built-in discrete-GPU prices represent **GPU-only prices**. These are not comparable to complete systems unless you enter host-machine cost and load/idle power. v3.1 asks for all three additional values before displaying full local-system TCO and payback. Enter zero only when the underlying GPU price or power data already include those contributions.

## Changes in v3.1

- Fixed incorrect maximum-context results for SWA-only and hybrid-attention architectures.
- Made fit/offload status and maximum-context calculations use the same placement function.
- Split maximum context into **no offload** and **with CPU offload**.
- Fixed double-counting of model parameters designated for system RAM.
- Fixed multi-GPU reserved capacity from one global allowance to one allowance per GPU.
- Included first-token decode time in TTFT.
- Separated GPU-only and complete-system cost and power inputs in TCO calculations.
- Restricted calibration to the measured configuration.
- Added backend-sensitive **compatibility warnings**, especially for native NVFP4/FP8 and quantized KV cache.
- Prevented unknown and missing model layer classes from silently underestimating KV cache by assuming unknown layers require full-attention KV; warnings still require user verification.
- Validated imported JSON and URL-shared scenarios, removed direct untrusted HTML interpolation, and excluded calibration data from shared links.
- Imported JSON is applied to the current page even where persistent local browser storage is blocked.
- Removed the unsupported generic claim of a ±20% error bound.

## Verification

- JavaScript syntax checked with Node.js.
- **2,156** model/hardware/context/concurrency combinations checked for consistency between `fit`, `offload` and their corresponding max-context limits.
- Additional focused unit tests for pure-SWA, hybrid-SWA, system-RAM parameter placement, first-token TTFT, GPU memory reservation, calibration isolation, unknown-layer parsing and JSON validation.
- Browser interactions tested with headless Chromium, including GPU host-cost inputs, quantization warnings, offload output, imported model data and an HTML-injection regression case.

## Important limitations

The displayed token/s and TTFT are analytical estimates, **not validated hardware benchmarks**. This calculator does not fully model attention kernels, actual KV dequantization costs, CPU-specific kernels, PCIe or multi-GPU communication, throttling, recurrent-model state for every architecture, batching effects, or scheduler queueing. It does not verify that a particular inference engine supports the selected quantization and model. Prices and quality rankings are editable snapshots, not live prices. Avoid relying on the estimates alone for hardware purchases or architecture decisions; compare against measured throughput on the intended hardware and runtime.
