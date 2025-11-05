# China AI Chips and Hardware Infrastructure Technology Panorama: Specifications, Performance, and Practical Optimization of Ascend/Cambricon/Enflame/Hygon DCU

## 1. Executive Summary and Core Conclusions

The training and inference of generative artificial intelligence have evolved from single-card capability comparison to a systems engineering competition involving "chip-card-node-cluster-network-software stack-job scheduling". Between 2024-2025, domestic AI chips made crucial strides in availability and large-scale deploment: Huawei's Ascend formed systematic solutions and toolchains around the 910B/910C, Cambricon utilized MLU370/MLU590 to cover both training and inference while strengthening multi-chip interconnectivity, Enflame extended from SuiSi training and YunSi inference to integrated training-inference and large-scale clusters, and Hygon DCU rapidly adapted mainstream large models with its "CUDA-like" ecosystem and full-precision training support. Compared to international advanced systems (NVIDIA A100/H100/H200/Blackwell), domestic chips still have gaps in interconnect bandwidth, memory subsystem, and software ecosystem maturity, but through engineering optimization and system-level collaboration, they have demonstrated practical cost-effectiveness and stable operational capabilities in specific workloads and domestic substitution paths.

From an application perspective, domestic chips show clear differentiation in comparative advantages and shortcomings: in communication-intensive Transformer training with extremely large parameter scales, inter-card interconnectivity (such as HCCS/MLU-Link/CUDA-like ecosystem compatibility solutions) and network topology (RoCE v2, InfiniBand, or custom switching) system-level optimization often have greater impact on training throughput and stability than single-card peak computing differences; on the inference side, KV Cache reuse, batch scheduling, and low-precision format (such as FP8/INT8) strategy choices directly determine the balance between latency and cost. Industry cases demonstrate that Ascend all-in-one solutions adapted to DeepSeek ecosystem, Cambricon's MLU370 deploment and energy efficiency performance in industry scenarios, Enflame's S60 inference cards achieving large-scale implementation in internet and intelligent computing centers, and Hygon DCU's comprehensive adaptation to mainstream domestic models are progressing from "pilot" (可用) to "production" (可管), but still require long-term refinement in cross-framework consistency, large-scale operations management, and complex fault tolerance strategies.[^5][^14][^16][^17][^19]

Key selection points and action recommendations can be summarized into three scenarios:

- Large-scale training and stable long-term operations: Prioritize evaluating system-level capabilities of inter-card interconnectivity and network topology, combined with the maturity of mixed-precision and communication-computation overlap strategies. The Ascend system demonstrates engineering advantages in multi-card collaboration and toolchains (MindSpeed, ACL, etc.); Cambricon's MLU590 is friendly to large model training in bandwidth and multi-card parallel capabilities; Hygon DCU reduces migration barriers with its "CUDA-like" ecosystem but requires targeted optimization based on target frameworks.[^2][^7][^18]
- Cost-optimized inference priority: Select platforms with greater low-precision and batch optimization space based on request latency and throughput targets. Enflame's S60 inference cards provide practical reference for "cost reduction and usability" through large-scale implementation in internet and multi-regional intelligent computing centers; Cambricon's MLU370 series demonstrates energy efficiency advantages in medium batch sizes suitable for continuous business iteration; Ascend solutions offer integration advantages in domestic substitution and software-hardware collaboration.[^13][^16][^17]
- Domestic substitution and ecosystem migration: Focus on migration costs of programming interfaces and libraries, adopting a four-step strategy of "operator coverage-compilation optimization-distributed communication-cluster governance". Cambricon's Neuware and Torch-MLU reduce PyTorch-side migration costs; Hygon DCU's ROCm ecosystem compatibility reduces adaptation complexity; Ascend shortens migration ramp-up periods through deeply adapted middleware and acceleration libraries.[^7][^8][^18]

Overall, domestic AI chips have achieved practical capabilities for undertaking some large model training and large-scale inference tasks, especially achieving stable benefits through system-level optimization under requirements for domestic substitution and controllable costs. Shortcomings are mainly concentrated in ecosystem maturity of high-bandwidth interconnectivity, visualization and automated governance of ultra-large-scale clusters, and operator consistency and performance reproducibility across frameworks. Looking ahead, FP8/FP4 low precision, Chiplet and advanced packaging, liquid cooling and optical interconnectivity will become important tools for domestic chips to bridge gaps.[^2][^4][^5]

## 2. Research Methodology and Data Sources

This study is based on official whitepapers and datasheets, vendor developer documentation, authoritative research reports, and in-depth media articles for cross-validation. Data timeliness is based on November 2025, with key sources including: NVIDIA A100/H100 architecture whitepapers and datasheets, Cambricon developer documentation and official product pages, Huawei Ascend documentation center and "Huawei Research" computing special issue, authoritative research reports from institutions like "东方财富证券" and "天风证券", as well as in-depth industry reports from EET-China and InfoQ.[^1][^2][^3][^7][^8][^14][^19]

Credibility principles prioritize official sources and whitepapers; media and blog data are marked with uncertainty and interpreted within ranges; missing fields or conflicting information are explicitly stated with risk warnings. This study also references industry annual and methodology reports to ensure consistency in metric specifications and terminology definitions.[^4]

Information gaps and uncertainty explanations: Detailed dual-Die packaging and precise peak computing specifications for Ascend 910C lack official datasheets, with existing data mainly from industry analysis and media reports; complete specifications for Enflame's fourth-generation L600/OGX (process, memory type/bandwidth, interconnect, power consumption) have limited public information, some from product releases and industry reports; complete training benchmarks for Hygon DCU ShenSuan-2/3 (FP16/BF16/FP32 matrix performance, multi-card interconnect topology) still lack authoritative systematic public data; some detailed items for NVIDIA Blackwell (B100/B200/GB200) are incomplete in public materials, only allowing range assessments based on architecture analysis and media reports. The above gaps are all noted with annotations in the text.[^5][^16][^17][^19][^20][^21]

## 3. Technical Foundation: Training/Inference Hardware and System Indicator Framework

Performance bottlenecks in large model training and inference depend not only on chip peak computing power and memory bandwidth but also on interconnect bandwidth, node topology, network topology, and software stack collaboration efficiency. To avoid "single metric distortion," a layered, reproducible indicator system and terminology specification need to be established.

- Computing power categories: Throughput capabilities of single-precision (FP32), half-precision (FP16), bfloat16 (BF16), and eight-bit integer (INT8) under different workloads, focusing on Tensor Core/matrix unit acceleration capabilities and effective benefits of sparsity/low precision.
- Memory and bandwidth: Card memory capacity and HBM type/bandwidth determine single-card model parameter and KV Cache capacity, also affecting gradient synchronization and activation overflow efficiency.
- Interconnect and network: Inter-card interconnectivity (such as NVLink, MLU-Link, HCCS) and inter-node network (RoCE v2, InfiniBand, Spectrum-X) topology selection and bandwidth/latency metrics directly determine communication-computation ratio and large-scale stability in distributed training.[^2][^4]
- Software stack and ecosystem: Significant impact of programming interfaces (CUDA, ROCm, proprietary SDKs), operator libraries, compilers, and distributed communication libraries (NCCL, etc.), as well as domestic platform acceleration libraries (MindSpeed, ACL, etc.) on training/inference efficiency and migration costs.[^8][^14]
- Precision formats and stability: Numerical stability and benefit differences of FP16/BF16/FP8 in training and inference need comprehensive consideration in system-level parameter tuning and loss scaling strategies.[^2][^4]
- System power consumption and cooling: Card and node TDP, thermal design power, and liquid cooling adaptation have veto power over large-scale cluster energy efficiency and PUE.[^4]

To help readers unify specifications, the following provides a terminology and indicator specification comparison table, followed by applied interpretations combined with specific chips and solutions in subsequent sections.

Table 1 Terminology and Indicator Specification Comparison Table (Example)

| Indicator/Term | Definition and Description | Focus Dimensions | Typical Impact |
|---|---|---|---|
| FP32/FP16/BF16/INT8 | Single-precision/half-precision/bfloat16/eight-bit integer data formats | Numerical stability and throughput | Affects training convergence and inference latency |
| HBM Capacity/Bandwidth | High-bandwidth memory capacity and access bandwidth | Model trainable scale, KV Cache | Affects training/inference throughput and overflow communication |
| Inter-card Interconnect (NVLink/MLU-Link/HCCS) | High-speed interconnect between cards | AllReduce/parameter synchronization efficiency | Affects communication-computation ratio and scalability |
| Inter-node Network (RoCE/InfiniBand) | Data center network interconnect technology | Bandwidth, latency, stability | Affects multi-machine multi-card throughput and job stability |
| Distributed Communication Library | Multi-machine multi-card communication implementation | Ring/Tree AllReduce, etc. | Affects gradient synchronization and parameter server modes |
| Low Precision/Sparsity | FP8/FP4 and structured sparsity | Effective computing power and bandwidth usage | Affects training speed and inference costs |

For intuitive display of interconnect ecosystem evolution, see the following illustration.

![NVLink Evolution Diagram (Source: Research Report/Whitepaper Illustration)](.pdf_temp/viewrange_chunk_1_1_5_1762321863/images/o158j8.jpg)

The illustration emphasizes the significance of NVLink's generational bandwidth improvements for large-scale multi-GPU system scalability, offering equally important insights for domestic platforms' inter-card interconnectivity and node network design.[^4]

## 4. Huawei Ascend: Technical Analysis and Practice of Ascend 910B/910C in Large Model Training

### 4.1 Architecture and Packaging

The Ascend 910 series is based on the DaVinci architecture, with 910B improving energy efficiency ratios through process and design optimization, and some versions (such as 910B3) introducing more advanced HBM3e with significant bandwidth improvements; 910C is viewed by the industry as integrating two 910B logic dies in the same package through advanced packaging to improve overall computing power density and production yield accessibility. This "dual-die integration" approach has engineering rationality in cost, yield, and supply speed, but detailed parameters (interconnect structure, cache coherence, etc.) still lack official datasheet support, requiring "range assessment" rather than "point estimate" usage of related data.[^5][^11][^12]

### 4.2 Technical Specifications and Interconnect

Based on public materials, key characteristics of Ascend 910B include: FP16 peak computing power approximately in the 280-320 TFLOPS range (with variations across versions and sources), some versions featuring 64GB HBM2e with approximately 400 GB/s bandwidth; 910B3 introduces HBM3e with bandwidth improved to approximately 1.2 TB/s, providing stronger support for trillion-parameter model training. In interconnectivity, Ascend provides HCCS high-speed interconnect (NVLink comparable) and PCIe 4.0, RoCE v2 and other network adaptation capabilities, providing paths for multi-card collaboration and inter-node communication. Estimated performance of 910C shows single-card FP16 reaching approximately 800 TFLOPS level, but given incomplete official specifications, careful evaluation of benchmarking relationships and large-scale stability is required.[^5][^11]

Table 2 Ascend 910B vs 910C Key Parameters (Range and Uncertainty Notes)

| Model | Process/Technology | Peak Computing (FP16) | Memory Capacity | HBM Type/Bandwidth | Interconnect | Power (TDP) | Notes |
|---|---|---|---|---|---|---|---|
| 910B | SMIC N+1 (7nm equivalent) | ~280-320 TFLOPS | 64GB | HBM2e / ~400 GB/s | HCCS, PCIe 4.0, RoCE v2 | ~310W (reported) | Performance variations across versions |
| 910B3 | SMIC N+1 | Same range | — | HBM3e / ~1.2 TB/s | Same as above | — | For trillion-parameter training |
| 910C | SMIC N+2 (7nm) | Estimated ~800 TFLOPS | HBM (varied sources) | — | HCCS, PCIe, RoCE | — | Dual-die packaging, official details not public |

Illustrative materials for architecture evolution reference:

![Ascend Ecosystem/Technology Illustration (Research Report Diagram)](.pdf_temp/viewrange_chunk_2_6_10_1762321864/images/f4kzse.jpg)

![NVLink/NVSwitch Evolution for Interconnect Comparison Inspiration (Research Report Diagram)](.pdf_temp/viewrange_chunk_1_1_5_1762321863/images/o158j8.jpg)

The above comparisons emphasize that in multi-card, multi-node large model training, interconnect topology and bandwidth often impact system throughput more than single-card peak differences. Ascend achieves "engineering practical" parallel efficiency under domestic conditions through HCCS and network stack collaboration, but still requires continuous optimization in large-scale consistency and fault tolerance governance.[^2][^4][^5]

### 4.3 Software Stack and Large Model Adaptation

The Ascend software stack includes compilers (similar to ACL levels), acceleration library MindSpeed, and end-edge-cloud toolchains. For large models, Ascend provides multi-dimensional acceleration algorithms including parallel optimization, memory optimization, communication optimization, and computation optimization to improve training speed and stability. At the practical level, Ascend's all-in-one solutions with DeepSeek ecosystem, combining software-hardware collaboration and domestic adaptation, demonstrate replicability in scenarios of "stable long-term operations, automatic fault recovery, and domestic substitution."[^14][^15]

### 4.4 Performance and Cases

Industry reports and solution analyses show that Ascend 910B/910C have been adopted by multiple leading enterprises for conferences and various AI workloads, with all-in-one solutions emphasizing "domestic, low-cost, software-hardware collaboration" integration capabilities. In actual migration, Ascend team's toolchain support and model adaptation are crucial for shortening ramp-up periods, especially in operator coverage and communication stack tuning.[^5][^19]

For more intuitive presentation of Ascend's benchmarking with other platforms, see the following performance comparison illustration.

![Ascend Performance Comparison Chart (Research Report Diagram, for Illustrative Comparison)](.pdf_temp/viewrange_chunk_2_6_10_1762321864/images/twkdw1.jpg)

The significance of the illustration is to remind readers that single-card comparison is just the starting point, with system-level tuning and ecosystem maturity being the main battlefield determining training/inference efficiency and total cost of ownership (TCO).[19]

### 4.5 Risks and Challenges

Main challenges in the Ascend ecosystem focus on: compatibility breadth and depth with CUDA/ROCm, network topology and link stability in large-scale clusters, cross-framework operator performance consistency, and operations management complexity. For enterprise users, migration costs and engineering experience in stable long-term operations often determine project success, recommending establishing a closed-loop methodology of "toolchain-operator coverage-distributed communication-monitoring governance."[^8][^19]

## 5. Cambricon: Technical Characteristics of MLU370/MLU590 and Large Model Training Optimization Strategies

### 5.1 Chip and Card Forms

Siyuan 370 adopts 7nm process and Chiplet architecture, being Cambricon's first Chiplet AI chip for unified training and inference platforms. Card forms cover MLU370-X4 (single-slot 150W, emphasizing high cost-effectiveness), MLU370-X8 (dual-core quad-chiplet, dual-slot 250W, for mid-to-high-end training/inference), and MLU370-S4/S8 (low power consumption, PCIe Gen4 compatible). MLU590 is positioned as training flagship, emphasizing large bandwidth and multi-card parallel capabilities.[^7][^8][^9][^10]

### 5.2 Technical Specifications and Parallel Capabilities

Public资料显示,MLU590(2024年6月发布)聚焦训练与推理,采用7nm工艺,具备约314 TFLOPS的FP16算力、80GB内存、约2048 GB/s带宽与约318.8 GB/s的互联带宽,最大设计功耗约350W。MLU370-X4提供约96 TFLOPS的FP16算力、24GB LPDDR5与约307.2 GB/s带宽,最大设计功耗约150W。寒武纪的MLU-Link与多芯互联技术面向多卡并行场景,结合自研软件栈Cambricon Neuware、MagicMind以及Torch-MLU,降低从PyTorch生态的迁移成本。[^6][^7][^8][^9][^10]

Table 3 MLU370-X4/X8/S4/S8 Key Parameters (Public Specifications)

| Model | Positioning | Process | Peak Computing (FP16/FP32/INT8) | Memory | Bandwidth | Power | Interconnect |
|---|---|---|---|---|---|---|---|
| MLU370-X4 | High Cost-Effectiveness | 7nm | ~96 TFLOPS / ~24 TFLOPS / 256 TOPS | 24GB LPDDR5 | ~307.2 GB/s | ~150W | PCIe Gen4, MLU-Link |
| MLU370-X8 | Mid-to-High End | 7nm | Dual-core stacked | 48GB (estimated) | — | ~250W | Same as above |
| MLU370-S4/S8 | Low Power | 7nm | Varies by model | — | — | ~75W | PCIe Gen4 |

Table 4 MLU590 Key Specifications (Public Specifications)

| Indicator | Value (approx.) | Description |
|---|---|---|
| FP16 Computing | 314 TFLOPS | Training/inference balanced |
| Memory Capacity | 80GB | Large model training friendly |
| Memory Bandwidth | 2048 GB/s | Bandwidth critical for throughput |
| Interconnect Bandwidth | 318.8 GB/s | Multi-card parallel capability |
| Max Power | 350W | Card design |

For understanding Cambricon's software stack levels, see the following illustration.

![Cambricon Software Stack Hierarchy Illustration (Research Report Diagram)](.pdf_temp/viewrange_chunk_1_1_5_1762321863/images/w48xlp.jpg)

This stack structure emphasizes deep coupling from bottom-layer Runtime to programming languages (BANG) and operator libraries, facilitating targeted optimization and performance reuse for different workloads.[^7]

### 5.3 Large Model Optimization Strategies and Ecosystem

Cambricon's optimization strategies for large models focus on:

- Parallel efficiency improvement: Optimizing multi-card parallel efficiency through MLU-Link and bridging solutions (such as four-card interconnect), combining AllReduce and pipeline parallel strategy combinations to improve communication-computation ratio.[^7]
- Software-hardware collaboration: Neuware provides bottom-layer operators and compilers, MagicMind optimizes inference paths, Torch-MLU supports direct usage in PyTorch ecosystem, significantly reducing migration barriers.[^7][^8]
- Multi-precision and numerical stability: Scenario-specific selection between FP16/BF16 and INT8, combined with loss scaling and gradient clipping strategies to ensure training stability.[^8]

### 5.4 Applications and Effectiveness

Industry scenario deployment cases show that MLU370, combined with medium-batch inference and energy optimization in workloads like medical imaging and industrial vision, demonstrates good performance/power/cost balance; in multi-card parallel training, stable throughput can be achieved under controllable power through operator coverage and communication optimization. In domestic substitution paths, Cambricon's ecosystem toolchains reduce migration costs from CUDA/ROCm to proprietary stacks.[^7][^13]

![Cambricon AI Chip Related Charts (Research Report Diagram)](.pdf_temp/viewrange_chunk_2_6_10_1762321864/images/5l02mo.jpg)

The illustration demonstrates Cambricon's full-stack capabilities and product line coverage, with this "from IP to cloud" systematic layout being key to providing sustainable support in domestic substitution.[^7]

### 5.5 Risks and Challenges

In advancing toward larger-scale training and more complex model structures, Cambricon's ecosystem still needs improvement: cross-framework operator performance consistency, congestion and packet loss governance in ultra-large-scale networks, and deep coupling with mainstream distributed frameworks and fault tolerance strategies all require continuous iteration and empirical data accumulation.[^7]

## 6. Enflame: Training and Inference Product Lineup, Technical Characteristics and Innovation Points

### 6.1 Technical Path and Product Evolution

Enflame started from SuiSi DTU training chips, launching YunSi T10 training cards and YunSi i10/i20 inference cards; in recent years, based on focus on inference market and integrated training-inference strategic judgment, released the new-generation S60 inference cards and introduced L600/OGX series emphasizing integrated training-inference architecture and domestic MaaS platform implementation. The company has achieved commercial deployment in training and inference scenarios across internet, finance, education and other fields, forming large-scale applications in multiple regional intelligent computing centers.[^13][^17]

### 6.2 Specifications and Innovation

SuiSi 2.5 adopts 12nm FinFET process, emphasizing improvement in transistor efficiency per unit area; S60 inference cards achieve "seventy thousand card scale" deployment verification in internet and multi-regional intelligent computing centers, demonstrating usability and cost advantages in inference scenarios. L600/OGX and other new product lines emphasize integrated training-inference and domestic platform fusion, targeting large-scale intensive application scenarios.[^16][^17]

Table 5 Enflame Product Overview (Training/Inference/Integrated Training-Inference: Public Information)

| Product/Chip | Positioning | Process | Main Characteristics | Application Scenarios | Scale Status |
|---|---|---|---|---|---|
| SuiSi DTU (YunSi T10) | Training | 12nm | Programmable, training acceleration | Data center training | Early commercial |
| YunSi i10/i20 | Inference | 12nm | High-bandwidth inference, scenario coverage | CV/NLP/recommendation, etc. | Continuous implementation |
| S60 Inference Card | Inference | — | Inference optimization | Internet/intelligent computing centers | Seventy thousand card scale |
| L600/OGX | Integrated Training-Inference | — | Domestic platform oriented | Integrated training-inference and MaaS | New product line |

Enflame's innovation points focus on: cost and energy efficiency on the inference side, verifying commercial feasibility of domestic computing through large-scale deployment; unified hardware foundation for different workloads through integrated training-inference product lineup; shortened scenario implementation cycles through domestic platforms and ecosystem collaboration.[^13][^16][17]

### 6.3 Risks and Challenges

Specification transparency and ecosystem toolchain completeness still need improvement; steady-state operations management of large-scale clusters and complex fault tolerance strategies are core challenges in scenarios emphasizing both inference and training. Enterprise selection should combine target frameworks and workload characteristics for sufficient verification and TCO assessment during PoC phases.[^13]

## 7. Hygon DCU: Parallel Computing Capabilities and Large Model Training Suitability

### 7.1 Architecture and Ecosystem

Hygon DCU is based on GPGPU architecture, adopting "CUDA-like" general parallel computing ecosystem compatible with mainstream international commercial and AI software, supporting full-precision (FP64/FP32/FP16) and integer data computing, targeting both HPC and AI training/inference. Hygon follows "CPU+DCU dual-wheel drive" strategy with server clusters/data centers as primary deployment forms.[^18][^21]

### 7.2 Training Suitability and Model Adaptation

Research reports show that Hygon DCU has achieved comprehensive adaptation and application with mainstream domestic large models including LLaMa, GPT, Bloom, ChatGLM, WuDao, and ZiDong Taichu, with ShenSuan-1 performance expected to reach over 40% of A100, and ShenSuan-2 released in Q3 2023 with expected performance doubling compared to ShenSuan-1. The company is advancing R&D and performance leapfrog of ShenSuan-3 to further address large model training and inference shortcomings.[^18][^20][^21]

Table 6 Hygon DCU ShenSuan-1/2 Public Key Points (Range Specifications)

| Model | Core Capabilities | Ecosystem Adaptation | Performance Specification | Deployment Form |
|---|---|---|---|---|
| ShenSuan-1 | Full-precision floating/integer | CUDA-like, mainstream compatible | Expected 40%+ of A100 | Server clusters/data centers |
| ShenSuan-2 | Performance doubled vs ShenSuan-1 (expected) | Continuously improving | Official details not fully public | Same as above |

![Domestic AI Chip Ecosystem Illustration (Research Report Diagram)](.pdf_temp/viewrange_chunk_1_1_5_1762321863/images/yhtxp2.jpg)

The illustration emphasizes multi-path evolution of domestic AI chip ecosystems, with Hygon reducing migration barriers through "CUDA-like" compatible paths as one of the practical tools for domestic substitution.[^18]

### 7.3 Risks and Challenges

Hygon DCU still needs more public data on authoritative benchmarks for large model training and multi-card interconnect topology details; deep coupling with mainstream distributed training frameworks and large-scale cluster governance practices also need more enterprise-level sample support. Enterprise selection recommends sufficient verification based on PoC and operator coverage and communication stack adaptation for target frameworks.[^18][^20]

## 8. International Benchmarking: NVIDIA A100/H100/H200/Blackwell (B100/B200)

NVIDIA A100 (Ampere architecture) provides approximately 19.5 TFLOPS FP32 and approximately 312 TFLOPS FP16, supporting third-generation NVLink with maximum bandwidth up to approximately 600 GB/s; H100 (Hopper architecture) significantly enhances new features like Transformer Engine and FP8, with FP16/BF16 matrix capability reaching approximately 1,979 TFLOPS, supporting fourth-generation NVLink (approximately 900 GB/s); H200 further improves memory capacity and bandwidth, adapting to ultra-large model inference and training scenarios; Blackwell (B100/B200/GB200) continues evolution in architecture and interconnectivity, with official and media materials showing system-level capability leapfrog in the AI factory era, but detailed items are still incomplete in public channels.[^1][^2][^3][^4][^5]

Table 7 A100/H100/H200/Blackwell Key Parameters (Public Specifications)

| Model | Architecture | Memory Capacity | Memory Bandwidth | FP16/BF16 Matrix | FP32 | Interconnect (NVLink) | Notes |
|---|---|---|---|---|---|---|---|
| A100 | Ampere | 80GB HBM2e | ~2,039 GB/s | ~312 TFLOPS | ~19.5 TFLOPS | Third-gen, ~600 GB/s | Datasheet and whitepaper |
| H100 | Hopper | 80GB HBM3 | ~3.35 TB/s | ~1,979 TFLOPS | ~67 TFLOPS | Fourth-gen, ~900 GB/s | Transformer Engine/FP8 |
| H200 | Hopper | 141GB HBM3e | ~4.8 TB/s | Improved vs H100 | — | NVLink/NVSwitch | Enhanced inference and HPC |
| Blackwell (B100/B200/GB200) | Blackwell | — | — | — | — | Continuing evolution | Detailed items incomplete |

![GPU Architecture Evolution Illustration (Research Report Diagram)](.pdf_temp/viewrange_chunk_2_6_10_1762321864/images/a7s5f8.jpg)

This evolution route highlights generational improvements in memory/bandwidth and interconnectivity, and the systemic impact of low-precision and sparsity strategies on AI workloads. Combined with domestic platform status, there are still periodic gaps in high-bandwidth interconnectivity and network ecosystems, but actual workload performance gaps can be narrowed through system-level optimization.[^1][^2][^3][^4][^5]

## 9. System-Level Comparison: Domestic AI Chips vs International Advanced Chips

Comparison should avoid the "single-card peak" trap, examining memory/bandwidth, interconnectivity and networking, software stack maturity, power consumption and cost, as well as ecosystem support and migration difficulty from system dimensions to draw conclusions with business guidance significance.

Table 8 System-Level Comparison (Ascend/Cambricon/Enflame/Hygon DCU vs A100/H100/H200/Blackwell)

| Dimension | Ascend (910B/910C) | Cambricon (MLU370/MLU590) | Enflame (S60/SuiSi/L600) | Hygon DCU (ShenSuan 1/2/3) | A100/H100/H200/Blackwell |
|---|---|---|---|---|---|
| Memory/Bandwidth | 910B3 introduces HBM3e, bandwidth ~1.2TB/s | MLU590 80GB, ~2048 GB/s | Limited public details | Full precision and "CUDA-like" ecosystem | H200 141GB HBM3e, ~4.8TB/s |
| Interconnect/Network | HCCS, RoCE v2 | MLU-Link and bridging | Focus on inference scale | Node cluster oriented | NVLink/NVSwitch/InfiniBand |
| Software Stack | Compiler/ACL, MindSpeed | Neuware, MagicMind, Torch-MLU | New product ecosystem maturing | CUDA-like compatible path | CUDA, NCCL, TensorRT mature ecosystem |
| Power/Cost | 910B ~310W (reported) | MLU370-X4 ~150W | S60 inference scale | Server cluster oriented | H100/H200 TDP and price ranges higher |
| Migration Difficulty | Improving domestic toolchain maturity | PyTorch backend and BANG language | Requires PoC verification | Lower migration threshold | Mature ecosystem, low migration cost |

![Global AI Server Market Size (Research Report Diagram)](.pdf_temp/viewrange_chunk_1_1_5_1762321863/images/w48xlp.jpg)

Combined with the market size illustration, we can see that international platforms' first-mover advantage and ecosystem maturity create a "snowball effect." To achieve competitiveness at the system level, domestic platforms must focus on overall optimization of "interconnect-network-software stack-operations" to form engineering usability advantages.[^4][^5][^13]

## 10. Optimization Strategies and Technical Challenges: Practical Methodology for Domestic Chips

Large model workload system optimization can be divided into four layers: parallel strategies, memory and communication, operators and frameworks, engineering and stability.

- Parallel strategy optimization: Combining data parallel (DP), tensor parallel (TP), and pipeline parallel (PP) strategies to optimize communication-computation overlap and intra/inter-batch scheduling, reducing AllReduce blocking on critical paths.
- Memory and communication: Reducing memory pressure through gradient accumulation and mixed-precision strategies, combined with distributed checkpointing and communication library optimization (such as Ring/Tree AllReduce) to improve multi-machine multi-card throughput.
- Operators and frameworks: Prioritizing coverage of key operators (such as attention, layer normalization, fused operators) on domestic platforms, using platform acceleration libraries (MindSpeed, ACL, etc.) and compilation optimization (such as Kernel Fusion, operator inlining) to reduce framework scheduling overhead.
- Engineering and stability: Establishing closed-loop monitoring governance for job fault tolerance, self-healing, and continuous training; in domestic substitution, constructing standardized processes from toolchain to automated operations management.

Table 9 Typical Large Model Memory Occupation Estimation (Example)

| Model | Precision | Parameter Scale | Weight Occupation (approx.) | Inference Addition (KV Cache, etc.) | Notes |
|---|---|---|---|---|---|
| Qwen2.5 72B | BF16 | 72B | ~140GB | ~20% weights | Slight variations by framework |
| R1-Distill-Qwen-32B | BF16 | 32B | ~64GB | ~20% weights | — |
| R1-Distill-Qwen-14B | BF16 | 14B | ~28GB | ~20% weights | — |
| R1-Distill-Qwen-7B | BF16 | 7B | ~14GB | ~20% weights | — |

![NVLink/NVSwitch Topology Insights for Communication Efficiency (Research Report Diagram)](.pdf_temp/viewrange_chunk_1_1_5_1762321863/images/o158j8.jpg)

The above table and illustration emphasize that in large model training, memory capacity and bandwidth determine "trainable scale," while interconnect topology and communication libraries determine "scalable efficiency." Domestic platform engineering optimization should follow the main thread of "operator coverage-communication optimization-batch scheduling-fault tolerance governance" to form reusable standard operating procedures (SOPs).[^2][^7][^14]

Main technical challenges include: insufficient ecosystem maturity of high-bandwidth interconnectivity, visualization and automated governance difficulties in ultra-large-scale networks, cross-framework operator consistency and performance reproducibility, and PUE optimization under liquid/air cooling hybrid conditions. Enterprises need to combine their own workloads and organizational capabilities to develop phased goals and measurement indicators.

## 11. Implementation Paths and Case Studies

- Ascend: All-in-one solutions combined with DeepSeek ecosystem, emphasizing software-hardware collaboration and domestic adaptation. Suitable for "stable training + rapid recovery" scenarios with toolchain and middleware acceleration capabilities.[^5][^14][^15]
- Cambricon: MLU370 deployed in medical imaging and industrial vision scenarios, combined with medium-batch inference and energy optimization, demonstrating practicality of training/inference balance.[^13]
- Enflame: S60 inference cards achieve "seventy thousand card scale" deployment in internet and multi-regional intelligent computing centers, verifying cost advantages and manageable stability on the inference side.[^17]
- Hygon DCU: Completed adaptation with mainstream domestic large models, relying on "CUDA-like" paths to reduce migration barriers, suitable for cluster solutions targeting HPC+AI integration.[^18]

Table 10 Case List (Example)

| Vendor | Scenario | Scale | Key Metrics | Migration/Optimization Points |
|---|---|---|---|---|
| Ascend | All-in-one + DeepSeek | Thousand-card level | Stable training/fault recovery | MindSpeed/ACL parallel and communication optimization |
| Cambricon | Medical imaging | Hundred-card level | Throughput/energy efficiency | MLU-Link multi-card and Torch-MLU migration |
| Enflame | Inference cluster | Ten-thousand-card level | Latency/cost | Batch processing and low-precision strategies, cluster governance |
| Hygon DCU | Domestic model adaptation | Hundred-card level | Compatibility/migration | CUDA-like path and framework operator coverage |

## 12. Conclusions and Roadmap Recommendations

- Training priority: For communication-intensive and ultra-large-scale parameter Transformer training, recommend prioritizing evaluation of system-level capabilities in inter-card interconnectivity and network topology, combined with maturity of mixed-precision and communication-computation overlap strategies. Ascend and Cambricon demonstrate engineering advantages in multi-card collaboration and toolchains; Hygon DCU can serve as a path supplement for reducing migration barriers.[^2][^7][^18]
- Inference priority: Recommend guiding by latency and throughput targets, combined with FP8/INT8 and batch scheduling strategy optimization, selecting platforms with large-scale deployment experience and operations toolchains. Enflame's S60 scale verification provides practical basis for "cost reduction and usability."[^17]
- Domestic substitution: Establish four-step methodology of "operator coverage-compilation optimization-distributed communication-cluster governance," phased assessment and measurement (such as training stability days, fault recovery time, throughput/energy consumption, etc.), avoiding unrealistic expectations of "one-step."[^7][^14]
- Medium to long-term layout: Track progress in FP8/FP4 low precision and sparsity, Chiplet and advanced packaging, liquid cooling and optical interconnectivity, advance talent and process reserves, forming "collaborative evolution" roadmap from chips to systems.[^2][^4][^5]

---

## References

[^1]: NVIDIA A100 Tensor Core GPU Architecture Whitepaper. https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf  
[^2]: NVIDIA H100 Tensor Core GPU Architecture Whitepaper. https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf  
[^3]: NVIDIA A100 Datasheet (简体中文). https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/a100/nvidia-a100-datasheet-nvidia-a4-2188504-r5-zhCN.pdf  
[^4]: 天风证券:计算机行业专题(2024-07-07). https://pdf.dfcfw.com/pdf/H3_AP202407071637650795_1.pdf  
[^5]: 一文看懂华为昇腾芯片(知乎专栏). https://zhuanlan.zhihu.com/p/1913660152676094004  
[^6]: 市场前景/规模预测/产业链及相关公司深度梳理(知乎专栏). https://zhuanlan.zhihu.com/p/1918004225083936970  
[^7]: 寒武纪官网. https://www.cambricon.com/  
[^8]: 寒武纪开发者社区文档中心. https://developer.cambricon.com/index/document/index/classid/3.html  
[^9]: 思元370系列 - 寒武纪. https://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=360  
[^10]: MLU370-X8智能加速卡 - 寒武纪. https://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=406  
[^11]: 国产AI芯片:华为昇腾的迭代路线(EDN China). https://www.ednchina.com/technews/36451.html  
[^12]: 昇腾910 AI芯片技术全面概述(与非网). https://www.eefocus.com/article/1840602.html  
[^13]: 国产AI算力行业报告:浪潮汹涌,势不可挡(2024-03-26). https://pdf.dfcfw.com/pdf/H3_AP202403261628250090_1.pdf  
[^14]: 《华为研究》计算专刊总第6期(2024). https://www-file.huawei.com/-/media/corp2020/pdf/publications/huawei-research/2024/huawei-research-issue6-cn.pdf  
[^15]: 昇腾DeepSeek一体机深度拆解(53AI). https://www.53ai.com/news/zhinengyingjian/2025042664092.html  
[^16]: 大模型落地提速国产算力受青睐(新华网). http://www.news.cn/tech/20240718/efbb7f2997e64d09b18c47daf95eab9b/c.html  
[^17]: 七万卡规模部署验证国产AI算力实力(智东西). https://m.zhidx.com/p/493741.html  
[^18]: 海光信息:国产算力领军企业,CPU+DCU双轮驱动(2024-08). https://pdf.dfcfw.com/pdf/H3_AP202408211639383896_1.pdf  
[^19]: 华为算力"公共事业":"超节点+全栈开源"如何撬动AI未来?(InfoQ). https://www.infoq.cn/article/gjh7qy46itf76nfiu9tx  
[^20]: 海光信息(688041.SH)跟踪报告(2024-05-14). https://pdf.dfcfw.com/pdf/H3_AP202405141633079130_1.pdf  
[^21]: 深度丨海光DCU与DeepSeek完成国产化适配(EET-China). https://www.eet-china.com/mp/a381638.html  
[^22]: 2024年中国AI大模型产业发展报告(人民网). http://download.people.com.cn/jiankang/nineteen17114578641.pdf