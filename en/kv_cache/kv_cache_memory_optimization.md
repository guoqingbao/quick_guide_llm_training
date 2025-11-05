# KV Cache Optimization and Memory Management Technology Research (2024–2025)

## Executive Summary

In 2024–2025, Large Language Model (LLM) services and training have entered a new normal of "long context + high concurrency + multi-tenancy": on one hand, 80k-level contexts and document/dialogue history-reuse-oriented business have made Time-to-First-Token (TTFT) and GPU memory capacity system bottlenecks; on the other hand, inference is constrained by KV Cache (Key-Value Cache) storage and bandwidth requirements that grow linearly with sequence length, while training faces the dual challenge of activation and gradient checkpointing computation-memory trade-offs. Around these core contradictions, the industry has formed systematic solutions including "quantization compression — paged management — memory pooling/fragmentation governance — checkpointing/SAC — SLO-adaptive transmission".

- Value and bottlenecks of KV Cache in inference. KV Cache saves and reuses attention module K/V tensors, transforming each generation step from "full recomputation" to "incremental computation + table lookup", essentially converting O(n²) attention redundant computation to O(n)-level new token computation and KV concatenation access, significantly reducing decoding latency; its cost is linear memory growth with context and becomes a network bottleneck when reused across machines/domains.[^1]
- 2024–2025 optimization technology spectrum and benefits. Four major technology routes form complementary solutions:
  1) KV quantization (e.g., INT4/INT2 with residual caching) can save approximately 2.5× memory while maintaining quality, extending 40k context of half-precision KV to 128k quantized KV on 80GB A100 (with Flash Attention), with INT2 requiring careful use;[^2]
  2) KV compression/streaming (CacheGen) forms end-to-end compact bitstreams through "Delta encoding + hierarchical quantization + arithmetic coding", reducing transmission surface by 3.5–4.3× and TTFT by 3.2–3.7× (relative to quantized baseline), even achieving 1.67–1.81× TTFT improvement compared to 8-bit "near-lossless";[^5]
  3) Paged KV management (vLLM PagedAttention/Hybrid KV manager) with non-contiguous storage, on-demand allocation, prefix sharing and near-zero internal fragmentation, significantly improving service concurrency and throughput with continuous batching;[^7][^8]
  4) Cross-level memory pooling and fragmentation governance (e.g., Pie's CPU memory pooling, STWeaver's spatio-temporal planning allocator, GLake's general memory/IO optimization) expand effective capacity and reduce fragmentation overhead from system level, achieving 1.27–1.9× end-to-end throughput improvement (Pie relative to vLLM), up to 32.5% throughput improvement (STWeaver), and up to 100% fragmentation reduction (STWeaver).[^12][^13][^14]
- Training-side checkpointing technology progress. PyTorch continues iterating on Selective Activation Checkpointing (SAC) and "memory budget API": on Transformers, recomputing only pointwise operations achieves approximately 50% activation memory savings, and combined strategies can keep the most expensive attention operations in the "save" path; SAC can sweep Pareto frontier in speed-memory space under different strategies. Empirically, gradient checkpointing often causes 20–50% single iteration slowdown but can exchange for 50–70% activation memory reduction and support larger effective batch sizes; Adacc, through "outlier-separated layered compression + activation checkpointing + MILP strategy scheduling", achieves 1.01–1.37× throughput improvement relative to baseline while maintaining accuracy (loss difference <0.5%), with maximum batch size up to 7.62×.[^9][^11][^16][^10]
- Practical implementation recommendations. For inference side, recommend combining KV quantization with PagedAttention/paged management, and introducing "CacheGen-style SLO-adaptive transmission" in cross-machine/cross-domain reuse scenarios; for training side, recommend using SAC/memory budget API for strategic recomputation, combined with STWeaver/GLake allocators to reduce fragmentation, and enabling Pie-style CPU expansion on nodes with sufficient CPU memory and high interconnect bandwidth (such as GH200) to improve effective capacity. For multi-tenant security, combine access control, isolation and audit mechanisms to prevent prompt leakage risks from KV sharing.[^15]

Key data excerpts:
- KV quantization (INT4+residual caching) can achieve FP16-level quality on Llama2-7B (Quanto INT4), with approximately 2.5× memory savings; compared to 8-bit near-lossless solutions, CacheGen still achieves 1.67–1.81× TTFT improvement;[^2][^5]
- vLLM achieves non-contiguous storage, dynamic allocation, and prefix sharing with PagedAttention, effectively eliminating static internal fragmentation and improving throughput and concurrency with continuous batching;[^7][^8]
- Pie spreads KV to CPU through "CPU memory pooling + concurrent prefetch exchange" on GH200, achieving up to 1.9× throughput relative to vLLM and latency reduction to 1/2, while maintaining near-compute SLO;[^12]
- STWeaver reduces fragmentation rate by up to 100% (average 79.2%), improves end-to-end throughput by up to 32.5%, with runtime overhead <0.05%;[^13]
- Training-side SAC and budget API achieve approximately 50% activation memory savings on Transformers; Adacc achieves 1.01–1.37× throughput improvement on V100 clusters, with maximum batch capacity improvement up to 7.62×.[^9][^10]

(Note: This paper systematically reviews and quantifies evidence from public papers and engineering documents in each section, with complete references in "References" at the end.)

---

## 1. Introduction and Research Scope

LLM inference and training are undergoing two types of structural changes. First, business is shifting from "short Q&A" to composite tasks of "long context + history reuse", making TTFT and GPU memory the primary bottlenecks; second, training scale-up and fine-tuning popularity make activation/gradient memory peaks and fragmentation issues increasingly prominent. This report focuses on the following objects and boundaries:
- Research objects: KV Cache (inference) and GPU memory management (inference/training), activation/gradient checkpointing (training), and extended cross-level memory pooling and streaming transmission impact on end-to-end SLO (Service-Level Objective).
- Time range: primarily 2024–2025 public papers and engineering documents, supplemented with necessary early background.
- Metrics: TTFT, throughput (tokens/s), GPU memory and fragmentation rate, maximum context length, batch size and training speed, quality impact (such as F1/accuracy/perplexity). Among these, TTFT still has differences in uniformity and comparability between industry and academia (different measurement approaches and noise sources), and this paper尽量标注环境与负载条件 when citing.[^5]
- Methodology stance: based on publicly verifiable materials, emphasizing complementarity and engineering trade-offs between methods; providing cautious interpretations for directions lacking unified benchmarks (such as fragmentation governance allocators).[^3][^5]

---

## 2. KV Cache Mechanism and System Role

KV Cache is essentially a "time-for-space" computation reuse mechanism. In the decoding phase of Transformer inference, attention requires historical K/V participation; KV Cache saves K/V tensors of all previous tokens, making each generation step only need to compute current token K/V and concatenate with cache for attention participation, thus avoiding redundant forward computation of historical sequences and significantly reducing computational burden of decoding steps.[^1]

The prefill and decode phases have different load characteristics: the former is compute-intensive, the latter is bandwidth-limited. Prefill is responsible for computing K/V for the entire prompt at once; decode generates tokens one by one, requiring repeated reading of historical K/V and constrained by GPU memory bandwidth. TTFT is jointly determined by prefill computation and network/storage transmission; when KV is reused across machines/domains, transmission becomes one of the primary latency sources.[^5]

KV Cache growth is positively correlated with parameters, number of heads, and context length. Taking Llama2-7B as an example: with sequence length 10k, 32 layers, 32 Key/Value heads, 128 dimensions per head, half-precision (2 bytes), KV Cache is estimated at approximately 5GB, close to one-third of model parameter GPU memory; in larger models or longer contexts, KV Cache easily reaches tens of GB or more, becoming the primary constraint for GPU memory expansion and TTFT.[^2][^5]

In implementation, avoiding memory fragmentation and allocation jitter caused by "repeated concatenation" is crucial. Common approaches are pre-allocating sufficiently large continuous tensors or using paged management, storing KV blocks as paging units at arbitrary locations in GPU memory, growing on-demand and sharing identical prefixes, fundamentally reducing internal fragmentation and improving concurrent reuse efficiency (vLLM PagedAttention).[^1][^8]

To intuitively show KV growth's impact on context limits, Table 1 provides a common engineering experience comparison (Hugging Face's measured approach on 80GB A100).

Table 1 Context Length vs KV Cache Estimation and Hardware Limits (Example: 80GB A100)

| Configuration                         | Supported Context (approx) | Description                         |
|---------------------------------------|-----------------------------|-------------------------------------|
| Half-precision KV Cache (FP16)        | 40k                         | Only KV occupancy approaches limit  |
| Quantized KV Cache (INT4/INT8) + Flash Attention | Up to 128k         | Depends on Flash Attention and quantization configuration |

The above data reflects: without changing model structure, INT4 quantization and Flash Attention collaboration can improve effective context limit from 40k to approximately 128k; however, quality and speed still need evaluation based on tasks and configurations (such as INT2's more significant quality degradation).[^2]

---

## 3. 2024–2025 KV Cache Optimization Technology Spectrum

Around "how to further compress KV size/transmission volume, reduce TTFT, and improve service concurrency without sacrificing quality", the industry has formed four complementary technology spectrums: quantization, compression/streaming, paging/sharing, and network/storage collaborative design. The following elaborates category by category, then provides comparison tables to highlight different methods' applicable boundaries and engineering trade-offs.

### 3.1 Quantization Class: KIVI, KIVI/Transformers (Quanto/HQQ), KVQuant, miniKV

Quantization compresses KV tensors by reducing numerical precision, with the core challenge being "how to maintain model quality under limited bits". Strategy-wise, it typically adopts:
- Residual cache: saving recent KV segments in original precision (e.g., default 128 tokens), quantizing early tokens, preserving precision for later tokens to reduce error propagation;[^2]
- Granularity selection: per-channel or per-token quantization, or KIVI's asymmetric scheme of "Key per-channel, Value per-token";[^2]
- Outlier handling: using higher precision for channels/elements significantly deviating from distribution, combined with layered/grouped quantization to reduce quality loss.

Representative works include: KIVI (asymmetric 2-bit scheme), KVQuant (quantization design for million-level contexts), Quanto/HQQ backend support in Transformers (INT2/INT4/INT8, device-agnostic), and miniKV's exploration of extremely low bit-width (2-bit) compression boundaries. Hugging Face reports show: on Llama2-7B, INT4 (Quanto) is close to FP16 quality, INT2 quality degradation is more significant; on 80GB A100, quantized KV combined with Flash Attention can improve context limit to approximately 128k (relative to FP16 KV's approximately 40k).[^2][3]

To illustrate quantization and Flash Attention's synergistic effect on context limits, Table 2 provides example-level comparison.

Table 2 Quantized KV and Context Limits (Example: 80GB A100)

| Configuration                         | Supported Context (approx) | Remarks                         |
|---------------------------------------|-----------------------------|---------------------------------|
| Half-precision KV Cache (FP16)        | 40k                         | Only KV occupancy               |
| Quantized KV Cache (INT4/INT8) + Flash Attention | Up to 128k         | Specific limit depends on quantization parameters and attention optimization |

This method class's advantage is simple implementation and low deploy的成本, suitable for "local GPU memory constrained but quality-stable" inference scenarios; the trade-off is: extremely low bit-width (such as INT2) quality risk increases, grouping strategy and residual length need tuning based on model and task.[^2][3]

### 3.2 Compression and Streaming: CacheGen (Delta + Hierarchical Quantization + Arithmetic Coding + SLO-adaptive)

CacheGen is "compression-oriented KV transmission volume and end-to-end TTFT reduction", proposing three key designs:[^4][^5]
- Delta encoding (relative to anchor tokens): utilizing locality between tokens, taking Delta of same group tokens relative to anchor tokens, with variance significantly lower than original values, easier to compress;
- Hierarchical quantization: based on the empirical rule that "shallow layers are more sensitive, deep layers have higher tolerance", using more conservative (higher bit) quantization for early layers, allowing larger quantization error for later layers;
- Arithmetic Coding: building symbol distributions per layer and channel for lossless coding, combined with CUDA acceleration and transmission-decoding pipelining, significantly reducing transmission and processing overhead.

At the streaming level, CacheGen divides context into multiple sub-blocks, encoding multiple quantization-level bitstreams offline per sub-block, online dynamically selecting "quantization level or falling back to text (recomputed by LLM)" based on bandwidth and SLO, and pipelining with previous sub-block decoding. In effect, compared to text context, CacheGen reduces TTFT by 3.1–4.7×; relative to default quantized baseline, TTFT reduction is 3.2–3.7×; even compared to 8-bit "near-lossless", still achieves 1.67–1.81× TTFT improvement; KV transmission volume reduction is 3.5–4.3×. Its quality impact is controlled within accuracy ≤2%, F1 decrease <0.1, perplexity decrease <0.1 across multiple datasets.[^5]

CacheGen methodology's value is: unlike "runtime tensor-form" KV compression, it takes "transmission-time bitstream" as optimization target, thus complementary to runtime technologies like KV quantization and token trimming, forming cumulative benefits under broader system boundaries.[^5]

### 3.3 Paging and Sharing: vLLM PagedAttention and Hybrid KV Manager

vLLM introduces operating system virtual memory concepts to GPU memory management, implementing with PagedAttention: non-contiguous storage, dynamic allocation, prefix sharing, and near-zero internal fragmentation; combined with "Continuous Batching" to dynamically add new requests between decoding steps, immediately release completed sequences, and preempt when necessary.[^8] This design essentially manages "sequence-growing KV blocks" as paging units, both improving memory utilization and enhancing throughput and latency stability in concurrent scenarios.[^7][^8]

In larger-scale and disaggregated serving scenarios, related engineering practices are extending prefix caching and KV transmission capabilities across multiple instances/nodes, forming "local/shared cache + global index-aware" deployment patterns, providing paths for cross-node reuse and cost optimization.[^8]

### 3.4 Network/Storage Collaboration: Disaggregated Service, Distributed Prefix Caching and Security

When KV is not local and needs cross-machine/cross-domain loading, network becomes the bottleneck. CacheGen responds to this change with "compression + streaming + SLO-adaptive", significantly improving TTFT and service stability.[^5] In multi-tenant environments, KV sharing brings new attack surfaces for prompt leakage, requiring reinforced isolation boundaries in sharing strategy, access control and auditing.[^15] For cluster-level global optimization, DynamoLLM一类 frameworks propose systematic scheduling and reconfiguration strategies from resource allocation, energy efficiency and SLO collaboration perspectives, providing methodological references for finding optimal cost/performance solutions in inference clusters.[^19]

---

## 4. KV Optimization Method Comparison and Selection Recommendations

Different methods have different trade-offs in "quality—memory—TTFT—engineering complexity" dimensions. Engineering implementation should combine load characteristics (context distribution, concurrency, bandwidth) with SLO to choose combined strategies. The following table compares four mainstream methods.

Table 3 KV Optimization Method Comparison (Qualitative Summary)

| Method                      | Compression/Savings Magnitude           | Quality Impact (Representative Observations)                              | TTFT Impact (Representative Observations)                   | Engineering Complexity/Dependencies                     | Typical Scenarios                                     |
|----------------------------|------------------------------------------|---------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------|--------------------------------------------------------|
| KV Quantization (INT4/INT2, etc.) | ~2.5× (INT4 example)       | INT4 close to FP16, INT2 quality degradation more significant (task-related) | Local compute benefits, cross-machine loading still constrained by transmission volume | Low (backends available), parameters need tuning | GPU memory constrained, context extension scenarios[^2] |
| KV Compression/Streaming (CacheGen)   | Transmission volume reduction 3.5–4.3× | Accuracy ≤2%, F1<0.1, perplexity<0.1 decrease (multiple datasets)         | TTFT reduction 3.2–3.7×; 1.67–1.81× relative to 8-bit     | Medium (CUDA AC, pipelining, SLO-adaptive)      | Cross-machine/cross-domain loading, long context, bandwidth fluctuation scenarios[^5] |
| Paging/Sharing (PagedAttention)| Near-zero internal fragmentation, improved concurrency | Quality unaffected (system optimization)                            | Significant throughput/concurrency improvement, more stable latency | Medium (framework integration, block management/continuous batching) | High-concurrency services, shared prefix, multi-request reuse[^7][^8] |
| Network/Storage Collaboration (Dynamo, etc.) | Cluster-level energy efficiency/cost optimization (system-level) | Quality unaffected (system strategy) | Depends on strategy, overall SLO more stable | High (cluster scheduling, disaggregated service) | Large-scale clusters, cross-region deployment, SLO-cost dual optimization[^19] |

Selection recommendations:
- If the main bottleneck is "local GPU memory + context length": prioritize KV quantization (INT4+residual cache), combined with Flash Attention to extend context limits; use INT2 cautiously on quality-sensitive tasks.[^2]
- If the main bottleneck is "cross-machine/cross-domain transmission TTFT": prioritize CacheGen-style compression + streaming + SLO-adaptive; complementary with runtime technologies like KV quantization/token trimming.[^5]
- If the main bottleneck is "concurrency and memory fragmentation": adopt vLLM PagedAttention and hybrid KV manager, combined with continuous batching; enable distributed prefix caching and sharing strategies under multi-instance deployment.[^7][^8]
- If seeking energy efficiency/cost and SLO balance at cluster level: refer to DynamoLLM-class frameworks for resource reconfiguration and global scheduling.[^19]

---

## 5. Memory Pool Management and Dynamic Allocation (GPU/CPU/Cross-level)

From a system perspective, KV optimization is not just "making data smaller", but also "making available capacity larger, minimizing fragmentation, and placing data at the correct hierarchy". Three paths are particularly key: CPU memory pooling (Pie), fragmentation governance allocators (STWeaver/GLake), and KV memory management references across frameworks (TensorRT-LLM).

### 5.1 CPU Memory Pooling: Pie (GH200)

Pie addresses "excessive KV occupancy on inference side", proposing "CPU memory pooling + performance-transparent exchange + adaptive expansion". Core approach: managing KV cache per layer, according to compute layer indices and mapping tables, swapping "cold layer" KV to CPU DRAM, prefetching to GPU HBM when needed; when expansion is insufficient or exchange latency exceeds compute latency, adjusting CPU allocation capacity online to ensure computation doesn't wait. On GH200 platform (900GB/s NVLink, CPU↔GPU 419/371 GB/s), relative to vLLM it achieves up to 1.9× throughput and latency reduction to 1/2; when maintaining same performance, GPU memory can be reduced to 1/1.67.[^12]

Key engineering points include: two-phase mapping updates (first allocate and temporarily store CPU pointers, switch mapping after exchange completion), FIFO strategy to maximize KV layer stay time on CPU, and dynamic reassignment modifications to block managers (rebuild continuous addresses and update free block lists during expansion/contraction). Since Pie relies on "concurrent exchange + high-bandwidth interconnect", deployment on platforms with similar interconnect capability (such as GH200) is recommended.[^12]

### 5.2 Fragmentation Governance: STWeaver and GLake

Fragmentation rate control directly relates to "effective capacity" and "allocation latency". STWeaver proposes a new paradigm of "offline planning + online allocation": through allocation analyzer recording requests' spatio-temporal characteristics, planning synthesizer groups requests by phase and size, static/dynamic dual-path planning, finally executed by runtime allocator, with priority allocation in dynamic reusable space. Empirical results show: average fragmentation reduction of 79.2% (up to 100%), end-to-end throughput improvement up to 32.5%, runtime overhead <0.05% compared to Vanilla PyTorch 2.3; on Qwen2.5-72B, maximum GPU memory saving of 56.3GB, on Llama2-7B reduction of 22.8GB (approximately 35.6%).[^13]

GLake targets "general optimization of GPU memory management and IO transmission", engineering-wise reporting "27% fragmentation reduction, 25GB GPU memory saving, training throughput improvement up to nearly 4× for 10B models". Though public materials focus on engineering summaries, its direction corroborates STWeaver: using system methods to improve memory utilization efficiency and reduce IO bottlenecks.[^14]

Table 4 Allocator/Memory Pool Method Comparison (Excerpts)

| Method         | Key Idea                         | Fragmentation/Memory Savings                         | Throughput Impact                    | Applicable Models/Loads             |
|----------------|----------------------------------|------------------------------------------------------|--------------------------------------|-------------------------------------|
| STWeaver     | Offline planning + online allocation, spatio-temporal pattern utilization | Fragmentation reduction up to 100% (average 79.2%)   | Up to +32.5%, runtime overhead <0.05% | Dense and MoE, stable training iterations[^13] |
| GLake        | General GPU memory/IO optimization | Fragmentation reduction 27%, 25GB saving (engineering report) | Training throughput up to nearly 4× (engineering report) | Training/Inference (general)[^14]     |
| Pie          | CPU pooling + concurrent exchange + adaptive expansion | GPU memory reduction to 1/1.67 (same performance)    | Relative to vLLM up to 1.9×          | Inference KV-dominant, high interconnect bandwidth[^12] |
| TensorRT-LLM | KV memory usage reference/common issues guide | —                                                  | —                                    | Engineering deployment reference[^18] |

### 5.3 Dynamic Allocation and Block Management: TensorRT-LLM, etc.

TensorRT-LLM provides engineering references for KV memory usage and common issues, helpful for estimating GPU memory before deployment, locating leaks and overflows, coordinating batch size and context length. Compared to vLLM's paged management, such tools provide practical guidance for "controllable GPU memory budget + efficient execution" deployment needs.[^18]

---

## 6. Activation Checkpointing and Gradient Checkpointing

Checkpointing technology's goal is very simple: using additional computation to exchange for lower peak memory. On Transformers and other large models, the implementation strategies and benefit/cost trade-offs of both differ significantly.

### 6.1 Activation Checkpointing (AC/SAC/Memory Budget API)

Traditional AC uses "checkpoint regions don't save intermediate activations, recompute during backward" as basic strategy, significantly reducing memory peak at backward start. PyTorch introduces two enhancement paths in versions 2.4/2.5:[^9]
- Selective Activation Checkpointing (SAC): using policy functions to decide which operators must be saved (such as matrix multiplication Attention), which can be recomputed (mostly pointwise operators), thus finding better points in speed-memory space;
- Memory Budget API (compile-only): users set memory budget (0–1), system automatically selects recomputation range; on Transformers, recomputing only pointwise operations achieves approximately 50% memory reduction, more recomputation means more memory decrease but more obvious speed degradation.

In engineering implementation, attention to operator cost differences across models, SAC strategy sensitivity to training speed, and compatibility with framework compilation stack is needed.

### 6.2 Gradient Checkpointing

Gradient checkpointing is one of the "oldest and most effective" memory-for-computation techniques: forward saves only partial activations, backward recomputes as needed. Experience shows it can reduce activation memory by 50–70%, but single iteration time increases by 20–50%; whether it brings overall training time decrease depends on whether "larger batch size + fewer iterations" combination can offset recomputation overhead. Reports from practical tuning show: on 8×A100 80GB, 8B model, enabling gradient checkpointing can increase per-GPU batch size from 2 to 12 (collapsible segments can reach 14), but overall training time doesn't exceed baseline without checkpointing, and when batch size increases to 14, performance shows "cliff", suggesting practical use should combine allocators and system bottlenecks for cautious expansion.[^11][^16]

### 6.3 Training-side Collaborative Optimization: Adacc

Adacc goes further at strategy level: performing "outlier separation + layered compression" on activations (Z-Score threshold 3 separates outlier channels; normal activations FP16→INT4), and using Mixed Integer Linear Programming (MILP) to solve optimal combination of "save, recompute, or compress" for each tensor; simultaneously providing "strategy correction" to adapt to dynamic changes in outlier distribution during training. Experiments show: relative to full recomputation and pure quantization, Adacc achieves 1.01–1.37× throughput improvement; maximum batch size improvement up to 7.62×; loss difference and quality impact controlled within <0.5% range.[^10]

Table 5 Training-side Checkpointing Technology Comparison (Excerpts)

| Method                 | Memory Savings                   | Throughput/Speed Impact                        | Quality Impact                | Applicability and Key Points                         |
|-----------------------|----------------------------------|------------------------------------------------|-------------------------------|------------------------------------------------------|
| AC (region-level)     | Medium (depends on checkpoint ratio) | Medium recomputation overhead                  | Essentially lossless          | Requires manual region division, easy to use[^9]     |
| SAC (selective)       | Can reach ≈50% (pointwise recomputation) | Depends on strategy: matrix multiplication/Attention tends to save | Essentially lossless          | Rich strategy space, needs tuning[^9]                |
| Memory Budget API     | Monotonic change with budget     | Minimizes recomputation under given budget     | Essentially lossless          | Available in compile mode, automated selection[^9]   |
| Gradient Checkpointing | 50–70% activation memory | Single iteration slowdown 20–50%               | Essentially lossless          | Focus on overall time and batch size trade-offs[^11][^16] |
| Adacc (checkpointing+compression) | Maximum batch capacity up to 7.62× | Throughput 1.01–1.37× | Loss difference <0.5% | Layered compression + MILP strategy, robust accuracy[^10] |

---

## 7. Performance and Engineering Balance: Decision Framework Oriented by SLO/TTFT/Throughput

LLM system optimization should return to the essence of SLO, TTFT and throughput. Recommend implementing a closed loop of "bottleneck profiling → strategy combination → parameter scheduling → runtime adaptation".

- Bottleneck profiling. Identify whether it's "compute-intensive (prefill)", "bandwidth-limited (decode)", "GPU memory-constrained (KV/activation peaks)" or "network-constrained (cross-machine KV transmission)"; combine context distribution, concurrency and bandwidth dimensions to form decision inputs.[^5]
- Strategy combination. Use KV quantization + paging management as foundation, overlay "CacheGen-style compression streaming" for cross-machine transmission; on training side use SAC/memory budget API and gradient checkpointing to control peak memory, combined with STWeaver/GLake to reduce fragmentation and expand effective capacity.[^2][^5][^7][^9][^13][^14]
- Runtime adaptation. Introduce SLO budget and bandwidth-aware transmission (such as CacheGen's block-level selection of quantization level or fallback to text recomputation), dynamically tuning parameters across model-engine-network three layers; consider Pie on nodes with sufficient CPU interconnect bandwidth to expand effective memory.[^5][^12]
- Evaluation metrics. Primarily end-to-end TTFT, throughput, maximum context and batch size, quality metrics (accuracy/F1/perplexity), while considering fragmentation rate, allocation latency and recomputation ratio.[^5][^13]

Table 6 Scenario-Strategy Mapping (Example)

| Scenario and Constraints                             | Recommended Combination                                        | Expected Benefits and Risk Alerts                                               |
|------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------|
| Single GPU memory constrained, medium context (<40k)       | INT4 quantization + residual cache + PagedAttention              | Memory savings ≈2.5×, quality stable; note INT2 quality risk and Flash Attention cooperation[^2][^7] |
| Cross-machine/cross-domain loading, long context (>40k)         | CacheGen compression streaming + quantized KV                      | TTFT reduction 3.2–3.7×, transmission volume reduction 3.5–4.3×; requires SLO adaptation and pipelining[^5] |
| High concurrency, shared prefix, multiple requests                | PagedAttention + continuous batching + prefix sharing           | Near-zero fragmentation, throughput/concurrency improvement; note multi-tenant isolation and auditing[^7][^8][^15] |
| High training activation peaks, severe fragmentation                | SAC/budget API + gradient checkpointing + STWeaver/GLake       | Activation memory -50~70%, fragmentation rate -79.2% (up to 100%); controllable recomputation overhead[^9][^11][^13] |
| GH200-class nodes, KV-dominant, capacity expansion needed           | Pie (CPU pooling + concurrent exchange)                         | Throughput up to 1.9×, latency to 1/2; interconnect bandwidth and mapping table updates need cautious tuning[^12] |

Table 7 Key Metrics Excerpts (Representative)

| Metric                         | Representative Data (environment-dependent)                                 | Source |
|-------------------------------|------------------------------------------------------|------|
| KV Quantization Memory Savings               | ≈2.5× (INT4 example)                                       | [^2] |
| KV Quantization Context Limit             | FP16 ~40k → quantization+Flash to 128k (A100 80GB)               | [^2] |
| CacheGen TTFT Improvement            | Relative to text 3.1–4.7×; relative to quantization 3.2–3.7×; relative to 8-bit 1.67–1.81× | [^5] |
| CacheGen Transmission Volume Reduction         | 3.5–4.3×                                                | [^5] |
| Pie Relative to vLLM Throughput/Latress         | Throughput up to 1.9×; latency reduction to 1/2                            | [^12] |
| STWeaver Fragmentation Rate/Throughput          | Fragmentation reduction up to 100% (average 79.2%); throughput up to +32.5%          | [^13] |
| Adacc Throughput/Batch Size             | Throughput 1.01–1.37×; maximum batch capacity up to 7.62×                       | [^10] |
| Gradient Checkpointing Memory/Speed          | Activation memory -50~70%; single iteration slowdown 20–50% (task-related)           | [^11][^16] |

The above metrics have limited reproducible experiments across different models, datasets, and network conditions, so engineering deployment needs A/B validation and capacity fallback design based on own SLO. [Information gaps see Section 10]

---

## 8. Practical Applications and Tool Ecosystem

- vLLM: centered on PagedAttention and continuous batching, combined with hybrid KV manager to achieve non-contiguous storage, dynamic allocation and prefix sharing, near-zero internal fragmentation, suitable for high-concurrency and shared-prefix online services.[^7][^8]
- PyTorch (AC/SAC/Memory Budget API): 2.4/2.5 provides selective recomputation and compile-time automated budget control, offering operable interfaces for training side to find Pareto frontier in "speed—memory" space.[^9]
- TensorRT-LLM: provides KV memory usage and common issues list, providing references for engineering deployment and capacity planning.[^18]
- CacheGen: open-source implementation provides end-to-end components of "KV encoding — streaming — adaptive", integrable with mainstream inference stacks to improve TTFT and bandwidth usage.[^17]

---

## 9. Risks, Security and Compliance

- Multi-tenant security. KV sharing, while improving reuse and throughput, may trigger prompt leakage risks; research shows in multi-tenant services, cache fragments of others can be read by constructing requests. Must implement access control, tenant isolation, encryption/anonymization and audit tracking, and carefully enable cross-tenant sharing strategies.[^15]
- Quality degradation and transparency. Compression/quantization brings precision fluctuation; engineering side needs monitoring and rollback mechanisms for quality drift; for SLO-adaptive strategies (such as CacheGen's block-level degradation), need to set "quality protection thresholds" to avoid significant quality fluctuation during network jitter.[^5]
- Resource preemption and isolation. Under continuous batching and preemptive scheduling, low-priority requests may face "starvation"; quotas, weights and fairness strategies need business SLO coordination to avoid service quality degradation.[^8]

---

## 10. Conclusions and Trends Outlook (So What)

Based on 2024–2025 evidence, KV optimization and memory management's "system solution" has taken shape: using quantization and paging management as foundation, compression streaming to alleviate transmission bottlenecks, memory pooling/fragmentation governance to amplify effective capacity, checkpointing technology to balance training-side speed-memory, supplemented by SLO-adaptive runtime strategies, to stably achieve TTFT and throughput goals under the new normal of "long context + high concurrency + multi-tenancy".[^3][^6]

Looking ahead to three main lines:
1) Hybrid optimization adaptation. Using "strategy learning + online tuning" to unify quantization, compression, paging, checkpointing and network transmission under SLO-driven cross-layer closed loop;
2) Software-hardware collaborative design. Under new GPU/CPU interconnect and storage systems (such as higher bandwidth inter-chip interconnect, hierarchical storage), redesigning KV layout, exchange granularity and pipelining;
3) Security-efficiency co-evolution. Constrained by confidential computing, encrypted caching and cross-domain sharing protocols, constructing coexisting mechanisms of "multi-tenant security + efficient reuse".[^6]

At the same time, key information gaps still need community collaboration to fill: unified quantization benchmarks across different models/frameworks (especially INT2/2bit) are still incomplete; fragmentation governance allocators lack cross-framework unified benchmarks; SAC/memory budget API generality evidence across different architectures and sequence lengths is insufficient; security auditing and isolation practices for cross-GPU/cross-machine KV sharing still need engineering validation; SLO-adaptive strategies in multi-tenant scenarios need more public data on robustness in complex network environments. [Information gaps synthesized from limitations of public papers and engineering documents, see discussions and future work sections of [^3][^5][^7][^8][^9][^13][^15]]

---

## References

[^1]: Sebastian Raschka. Understanding and Coding the KV Cache in LLMs from Scratch. 2025. https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms  
[^2]: Hugging Face. Unlocking Longer Generation with Key-Value Cache Quantization. 2024. https://huggingface.co/blog/kv-cache-quantization  
[^3]: A Review on Methods to Optimize LLM's KV Cache Consumption. 2024. https://arxiv.org/html/2407.18003v4  
[^4]: CacheGen: KV Cache Compression and Streaming for Fast LLM Serving. SIGCOMM 2024. https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf  
[^5]: CacheGen (ACM DOI). 2024. https://dl.acm.org/doi/10.1145/3651890.3672274  
[^6]: A Survey on Large Language Model Acceleration based on KV Cache Management. 2024. https://arxiv.org/html/2412.19442v3  
[^7]: Hybrid KV Cache Manager — vLLM Docs. https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html  
[^8]: Why vLLM is the best choice for AI inference today. Red Hat Developers. 2025. https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today  
[^9]: PyTorch Blog. Current and New Activation Checkpointing Techniques in PyTorch. 2025. https://pytorch.org/blog/activation-checkpointing-techniques/  
[^10]: Adacc: Adaptive Compression and Activation Checkpointing for LLM Training. 2025. https://arxiv.org/html/2508.00806v1  
[^11]: Giles Thomas. Fine-tuning LLMs — Gradient Checkpointing. 2024. https://www.gilesthomas.com/2024/09/fine-tuning-9  
[^12]: Pie: Pooling CPU Memory for LLM Inference. 2024. https://arxiv.org/html/2411.09317v1  
[^13]: STWeaver: Reducing GPU Memory Fragmentation via Spatio-Temporal Planning. 2025. https://arxiv.org/html/2507.16274v1  
[^14]: GLake: optimizing GPU memory management and IO transmission. 2024. https://github.com/antgroup/glake  
[^15]: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving. NDSS 2025. https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf  
[^16]: Efficient Training on a Single GPU — Transformers Docs. https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one  
[^17]: CacheGen GitHub Repository. https://github.com/UChi-JCL/CacheGen  
[^18]: Memory Usage of TensorRT-LLM — Reference. https://nvidia.github.io/TensorRT-LLM/reference/memory.html  
[^19]: DynamoLLM: Designing LLM Inference Clusters for Performance and Energy SLOs. 2024. https://arxiv.org/html/2408.00741v1

---
