# World Model Architecture From Kosmos Creators' Insights

**Andrew White's blog reveals the design principles that made Kosmos possible - structured world models that overcome coherence limits through persistent, queryable memory across hundreds of agent actions. This represents the culmination of a year-long evolution from cost economics to reasoning breakthroughs to production-ready AI scientist systems.**

The most critical insight: **structured world models are not just expanded context windows**. They're queryable databases enabling information from early discovery cycles to remain accessible after tens of millions of tokens, allowing coherent pursuit of research objectives over 200+ agent rollouts. This architectural innovation, specifically designed by Andrew White for Kosmos, solves the coherence problem that limited all previous AI scientist attempts to shallow, context-bound investigations.

Across five blog posts spanning October 2024 to October 2025, White's thinking evolved from foundational economics (discovery costs averaging $100M, doubling every 12 years) through reasoning model breakthroughs (small 7B models matching frontier 500B+ models on specialized tasks) to a precise, demanding definition of AI scientists that even his own systems don't yet fully meet. The journey reveals not just what Kosmos *is*, but why certain architectural choices became inevitable as the team confronted real-world discovery automation challenges.

## The evolution from Robin to Kosmos architecture

White's year-long journey crystallizes three major inflection points that shaped Kosmos's design. In October 2024, **economic analysis** established the financial imperative: with discovery costs doubling every 12 years and reaching $100M average in 2024, even modest automation efficiency gains justify substantial AI investment. His analysis spending $75 in API costs to evaluate 350+ discoveries across human history demonstrated extraordinary ROI potential - computational analysis scales linearly while human discovery costs scale exponentially.

By January 2025, **reasoning model breakthroughs** transformed the possible. FutureHouse's experiments with DeepSeek r1 and related systems revealed that simple strategies (generate chains of thought, sample multiple times, majority vote) dramatically outperformed complex approaches like Monte Carlo tree search. More critically, **small specialized models (7B-14B parameters) trained on successful demonstration trajectories could exceed frontier models 500x their size** on domain-specific reasoning tasks. This validated a fundamental architectural principle: separation of knowledge storage from reasoning capability, enabling world models to combine external knowledge retrieval with small, specialized reasoning engines.

The February 2025 update documented rapid commoditization - 27 reasoning models emerged in 30 days, with distillation achievable for under $100 in GPU compute. Three distinct training recipes crystallized: cold-start pure RL (requiring 32B+ parameters), distillation via supervised fine-tuning on reasoning traces (no size limits), and hybrid distillation plus RL (best results). The Deepscaler-1.5B model matching DeepSeek V3's performance (500x larger) on mathematical reasoning proved the knowledge-reasoning separation wasn't theoretical - it was implementable at production scale.

By October 2025, White synthesized these insights into Kosmos's definitive architecture and an uncompromisingly rigorous definition: **"An AI Scientist is a system whose input is a general direction of discovery and whose output is experimental results, analysis, and a paper describing a novel discovery."** Critically, White admits "I know of no AI Scientists based on my definition" - even Kosmos, which achieves 79.4% factual accuracy and performs 6-month equivalents of human researcher time per 20-cycle run, hasn't yet completed the full autonomous discovery cycle from quest to peer-reviewed paper.

## Core design principles extracted from the blog corpus

### 1. Structured world models enable persistence beyond context limits

**Evidence**: "Unlike a plain context window, it is queryable and structured, so information from early steps remains accessible after tens of thousands of tokens. The core innovation in Kosmos is our use of structured world models, which allow us to efficiently incorporate information extracted over hundreds of agent trajectories."

**Principle**: World models must be databases, not expanded contexts. Structure as entities, relationships, experimental results, and open questions - queryable by agents throughout multi-cycle runs. This architecture enabled Kosmos to coherently pursue objectives over 200 agent rollouts spanning ~42,000 lines of code and 1,500 papers, an order of magnitude beyond context-bound systems.

### 2. Simplicity beats complexity in reasoning architectures

**Evidence**: "DeepSeek r1's simple inference strategy (generate CoT → sample multiple times → majority vote) dramatically outperforms complex approaches. FutureHouse found it hard to efficiently do MCTS and found multiple times that predicting Q-values is worse than just fine-tuning on successful trajectories."

**Principle**: Avoid over-engineering reasoning control. Simple generate-sample-vote strategies work; Monte Carlo tree search with Q-value predictors adds complexity without proportional gains. The $75 discovery cost analysis itself exemplified this - straightforward retrieval-distillation-reasoning pipeline outperformed sophisticated alternatives.

### 3. Small specialized models exceed large generalist models

**Evidence**: "Llama-3.1 7B beat Sonnet 3.6 when trained on successful demonstrations. DeepSeek r1 14B models beat o1-mini across multiple benchmarks. Deepscaler-1.5B supposedly better at math than Deepseek's V3 - about 500x larger."

**Principle**: Domain-specific small models (7B-14B) trained on successful trajectories achieve superhuman performance on specialized tasks. Separate knowledge storage (external databases, retrieval systems) from reasoning capacity (small model weights). This enables cost-effective scaling and explains Kosmos's architecture combining structured world models with specialized reasoning agents.

### 4. Dynamic model selection based on task complexity

**Evidence**: "I probably could have used o1-mini for everything pre-1950, because the estimates were necessarily crude. I did see a real difference for the more complex projects, where o1-mini gave $100M for the capital cost with no justification."

**Principle**: Route tasks to appropriate reasoning capacity. Simple historical estimates don't require expensive models; complex multi-component problems demand sophisticated reasoning. World models should implement tiered reasoning with selection logic routing by complexity, optimizing cost-benefit continuously.

### 5. Validation through ground truth anchoring is non-negotiable

**Evidence**: "The Human Genome Project was reported to have cost $2.7 billion when complete in 2003 ($4.5 billion in 2024 dollars). This is indeed the value returned from this method. Independent scientists found 79.4% of statements in Kosmos reports to be accurate."

**Principle**: Calibrate on known examples before deploying on novel problems. Maintain validation datasets establishing baseline accuracy. White's cost analysis validated against Human Genome Project and Hooker Telescope before extrapolating; Kosmos reports underwent rigorous external expert evaluation showing 85.5% accuracy on data analysis, 82.1% on literature statements, but only 57.9% on synthesis - revealing synthesis as the weakest capability requiring further development.

### 6. Complete provenance is mandatory for scientific credibility

**Evidence**: "Every statement traceable to specific Jupyter notebook cells or literature passages. Explicit provenance is important in scientific settings because it allows human collaborators to audit individual claims. Scientific conclusions must be grounded in fact, but too many AI systems produce conclusions without clear provenance."

**Principle**: Build auditability into architecture, not as post-hoc feature. Kosmos's parallel agents (data analysis, literature search) both write structured outputs to the world model with complete citation chains. Every conclusion must trace through reasoning to primary evidence. This transparency enables trust and enables identifying where synthesis errors occur.

### 7. Demonstration trajectories are more valuable than training data

**Evidence**: "Demonstration reasoning trajectories are fungible between models and model size, and may be more important in the long run relative to models or training data. They're like a power-up mushroom for normal base models, providing huge gains from just a little fine-tuning."

**Principle**: Invest in generating successful demonstration trajectories, not massive pre-training or human annotation. Trajectories from one model can bootstrap others; small models fine-tuned on under 1,000 successful traces achieve frontier performance. This insight informed Kosmos's architecture - the system *generates* successful reasoning trajectories during operation, creating a self-improving feedback loop.

### 8. Parallel execution with shared memory beats sequential chaining

**Evidence**: "Data Analysis Agent writes and executes code; Literature Search Agent retrieves papers - both access same world model. System proposes up to 10 concrete tasks per cycle based on research objective and current world model state."

**Principle**: Multiple specialized agents operating in parallel on shared world model enables cross-pollination impossible in sequential pipelines. Kosmos runs 20+ cycles of parallel literature search and data analysis, each updating the persistent world model, enabling pattern connection across datasets and disciplines. This contrasts with earlier systems where agents operated sequentially, losing context and coherence.

### 9. Epistemic honesty about uncertainty and limitations

**Evidence**: "Obviously this is unjustified, but we can see some examples. I feel sorry for o1-preview, but I can force it to speculate on the cost of inventing numeral zero. The longer the Kosmos run, the more likely it is that Kosmos descends down a rabbit hole, chasing spurious statistical correlations."

**Principle**: Explicitly flag confidence levels degrading with data sparsity. White consistently acknowledges system limitations - synthesis accuracy at 57.9%, tendency toward spurious correlations in long runs, and admission that no system (including Kosmos) yet meets his rigorous AI Scientist definition. Build uncertainty quantification into world model structure, not just outputs.

### 10. Language models as universal scientific interface

**Evidence**: "Language is the only way to connect protocols, analysis code, hypotheses, and scientific literature. It was not possible to build a generalized AI Scientist until language models reached the performance they have now."

**Principle**: Language serves as the integration layer connecting disparate scientific tools, databases, literature, and experimental protocols. This insight shaped FutureHouse's entire architecture - specialized agents (Crow for literature, Finch for data analysis, Phoenix for chemistry) all communicate through natural language interfaces, enabling composition and reducing infrastructure overhead compared to rigid programmatic APIs.

### 11. Iterative cycles with world model updates enable progressive refinement

**Evidence**: "Runs for up to 12 hours, performs 20+ cycles of: literature search → hypothesis generation → data analysis → world model update. Each cycle generates concrete, structured outputs written back to world model."

**Principle**: Single-pass systems are inherently limited. Discovery requires iterative refinement where each cycle builds on previous findings stored in persistent world model. Kosmos's architecture embeds this directly - task proposals generated from current world model state, agent outputs update world model, next cycle begins from enriched state. This creates a compounding knowledge effect impossible in stateless systems.

### 12. Challenges as assets more valuable than models or data

**Evidence**: "Challenging and meaningful problems are the most important asset. Labelers should be making hard problems, rather than annotating or 'preferencing'. You need a verifiable problem with ground truth."

**Principle**: Invest in problem generation and evaluation environment design over model scaling or data collection. White's reasoning post emphasized this shift: with bootstrapping from successful trajectories, the bottleneck moves from training data to meaningful problems with verifiable solutions. World models should interface with rich evaluation environments enabling continuous problem generation, not just knowledge retrieval.

### 13. Cost-benefit optimization guides architectural decisions

**Evidence**: "It cost me around $75.00 of API costs to do this whole calculation - 99% o1-preview. Discovery costs $100M and doubles every 12 years. For under $100 of GPU compute you can distill reasoning traces into small language models."

**Principle**: Every architectural choice should consider computational economics. World models should track their own costs, compare computation expense to problem value, and optimize resource allocation dynamically. White's analysis revealed that with discovery costs rising exponentially and AI costs falling dramatically, even modest efficiency gains from automation create extraordinary ROI - justifying sophisticated world model infrastructure.

### 14. Non-reasoning tasks require specialized representations

**Evidence**: "ProtocolQA (reasoning task): Performance increases with better models. CloningScenarios (DNA sequence analysis): No gain with better reasoning models - requires tools or specialized tokens. There are tasks actually not reasoning limited, but limited by some other attribute."

**Principle**: Not all scientific tasks benefit from reasoning. Some require specialized tools, representations, or domain-specific tokens. World model architecture must support heterogeneous agent types - reasoning agents for hypothesis generation, tool-use agents for sequence analysis, simulation agents for physical predictions. Recognize task type boundaries and route appropriately.

### 15. Steering through "quests" prevents meaningless discovery

**Evidence**: "Science is ultimately rooted in goals of society, otherwise we may end up with more digits of pi or a treatise on shrimp physiology. An AI Scientist system whose input is a general direction of discovery."

**Principle**: Pure autonomous exploration produces technically-novel but useless discoveries. Require human-specified "quests" - general directions of investigation rooted in societal goals. This distinguishes from specific hypotheses (which the system generates) and prevents computational resources from optimizing for arbitrary novelty. Kosmos's input is a research objective; outputs are discoveries advancing that objective.

## Recommended world model architecture with justifications

Based on White's insights, the optimal world model architecture combines structured persistence, modular specialization, and iterative refinement:

### Core components

**Structured Database Layer**: Not expanded context windows, but queryable graph databases storing entities (genes, proteins, compounds, experimental conditions), relationships (activation, inhibition, correlation), experimental results with metadata, and open questions. Use Neo4j-style property graphs enabling relationship traversal and pattern matching. **Justification**: White explicitly designed Kosmos's world model as structured and queryable; his discovery cost analysis showed retrieval-based systems (paper-qa) outperform monolithic approaches; reasoning posts emphasized separating knowledge storage from reasoning capacity.

**Specialized Agent Swarm**: Deploy multiple small models (7B-14B parameters) optimized for specific scientific workflows - literature search agents, data analysis agents, hypothesis generation agents, synthesis agents. Each agent reads from and writes to the shared world model. **Justification**: White's reasoning experiments proved small specialized models exceed large generalist models; FutureHouse's agent portfolio (Crow, Finch, Phoenix) validates specialization strategy; parallel execution with shared memory enables cross-pollination Kosmos demonstrated.

**Iterative Cycle Manager**: Orchestrates 20+ cycles of (1) task proposal based on world model state, (2) parallel agent execution, (3) structured output writing, (4) world model update. Each cycle operates on enriched state from previous cycles. **Justification**: Kosmos runs 20+ cycles enabling progressive refinement; White's discovery cost analysis used layered architecture (retrieval → distillation → reasoning); single-pass systems hit coherence limits that world models overcome.

**Complete Provenance System**: Every entity, relationship, and conclusion in world model links to originating evidence - specific code cells, literature passages, or experimental data. Implement as metadata on all database entries. **Justification**: White emphasizes provenance as non-negotiable for scientific credibility; Kosmos's 79.4% accuracy validated through tracing statements to sources; enables identifying synthesis weaknesses (57.9% accuracy) for targeted improvement.

**Dynamic Reasoning Router**: Selects model capacity (fast/cheap vs. slow/expensive) based on task complexity. Simple retrieval uses efficient models; complex multi-component synthesis uses premium reasoning. **Justification**: White's discovery cost analysis showed o1-mini failed on complex projects requiring o1-preview; reasoning posts validated that small models work for domain-specific tasks while general reasoning needs more capacity; cost-benefit optimization is core principle.

**Validation and Calibration Module**: Maintains ground truth datasets for continuous calibration; requires external expert evaluation; tracks accuracy by statement type (data analysis, literature synthesis, cross-domain synthesis). **Justification**: White validated discovery costs against Human Genome Project before extrapolating; Kosmos underwent rigorous external evaluation revealing synthesis as weakest capability; epistemic honesty demands continuous validation.

### Information flow architecture

**Input**: Research "quest" (general direction) plus relevant datasets, literature corpus access, and experimental constraints. **Processing**: Cycle 1 proposes initial tasks exploring quest; parallel agents execute, writing structured findings to world model. Cycle 2 analyzes world model state, identifies gaps/contradictions, proposes refined tasks. Process iterates 20+ cycles. **Output**: Synthesis agent traverses world model generating fully-cited report with experimental results, analysis, and contextual discussion.

The world model serves as both persistent memory and coordination mechanism - agents communicate indirectly through structured reads/writes rather than direct message passing. This enables asynchronous parallel operation and maintains coherence as agents accumulate tens of millions of tokens across cycles.

### Technology stack implications

**Graph Database**: Neo4j or equivalent for relationship-rich entity storage. **Vector Store**: Separate vector database (ChromaDB, Pinecone) for semantic literature search. **Code Execution**: Jupyter kernel environments for data analysis agents. **Model Serving**: Separate endpoints for different capacity tiers (7B specialized, 14B hybrid, frontier for complex synthesis). **Orchestration**: Cycle manager coordinating task generation, agent dispatch, and world model updates.

## Cost model and economic analysis

White's blog posts enable construction of a comprehensive cost model for world model operations:

### Capital costs (one-time setup)

**World Model Infrastructure**: Graph database hosting, vector store setup, code execution environment - estimated **$5,000-15,000** setup assuming cloud deployment. **Model Fine-Tuning**: Distillation of specialized small models costs under **$100 per model** per White's February analysis; assume **$500 total** for initial agent swarm (literature, analysis, synthesis, hypothesis agents). **Validation Datasets**: Curating ground truth for calibration - primarily human time, estimate **$2,000-5,000** for domain-specific examples.

**Total Capital**: **$7,500-20,500** one-time investment - dramatically less than traditional lab equipment.

### Operational costs (per discovery cycle)

**API Costs**: White's discovery analysis cost $75 for 350 discoveries = ~$0.20 per analysis. Kosmos runs at **$200 per 20-cycle run** (heavily discounted pricing). At scale: estimate **$10-50 per cycle** depending on model selection and depth. With 20 cycles per complete run: **$200-1,000 per discovery investigation**.

**Compute Costs**: Small model inference (7B-14B) extremely cheap at scale. Literature search via RAG similarly low-cost. Expensive component is synthesis using frontier models. Estimate **$50-200 per run** for synthesis operations.

**Storage Costs**: World model database grows with each investigation. Estimate **$10-50/month** for typical research group workload (10-20 investigations/month). Vector store for literature corpus more expensive initially but amortizes across investigations - **$100-500/month** depending on corpus size.

**Human Validation**: External expert evaluation of outputs. Kosmos evaluation involved assessing 102 statements from 3 reports. Estimate **4-8 hours expert time per validation** = **$400-1,200** per major finding validation at academic rates.

**Total Operational Cost Per Discovery**: **$660-2,450** including validation, or **$200-1,000** for system operation alone.

### Return on investment analysis

White's analysis showed average major discovery costs **$100M** with **12-year doubling time**. Kosmos performs **6-month equivalents of human researcher time** per run. Academic postdoc fully-loaded cost: ~$80,000/year = **$40,000 per 6 months**.

**Cost Savings Per Kosmos Run**: $40,000 (human equivalent) - $1,000 (system operation) = **$39,000 saved**. **ROI**: 3,900% per investigation. At scale with hundreds of investigations, capital costs amortize rapidly and operational costs scale linearly while human costs remain fixed.

### Optimization strategies

**Tiered Reasoning**: Route 80% of tasks to cheap small models (7B), 15% to mid-tier (14B), 5% to frontier models for complex synthesis. Reduces API costs 60-80% versus uniform premium model use.

**Cached Retrieval**: Store literature search results and common analyses in world model for reuse across investigations. White's paper-qa approach exemplifies this - retrieval cost amortizes.

**Batch Processing**: Run multiple investigations simultaneously sharing literature corpus and common world model infrastructure. Reduces marginal cost per investigation.

**Progressive Depth**: Start with shallow 5-cycle exploration ($100-250) before committing to deep 20-cycle investigation. Enables early validation before full resource commitment.

**Academic Partnerships**: White offers free tier for academics, recognizing that validation and trajectory generation from expert use creates value for system improvement. Consider similar model where early users subsidize development through feedback.

## Implementation roadmap informed by blog evolution

White's year-long journey from cost analysis to production Kosmos suggests a phased implementation approach:

### Phase 1: Foundation (Months 1-3) - Economics and retrieval

**Build**: Basic retrieval infrastructure (literature search, paper-qa integration), cost tracking and optimization framework, simple world model prototype (document store, not graph database). **Validate**: Reproduce White's discovery cost analysis ($75 for historical analysis), confirm order-of-magnitude reasoning works, establish accuracy baselines. **Priority**: This phase establishes economic justification and validates that LLMs can automate previously manual intellectual work. White started here in October 2024, testing "scaling up intelligence" with clear problem definition plus capable models.

### Phase 2: Reasoning specialization (Months 4-6) - Small models and distillation

**Build**: Fine-tune small specialized models (7B-14B) on successful demonstration trajectories for domain-specific reasoning, implement tiered model selection routing by complexity, expand world model to structured entities and relationships. **Validate**: Reproduce small model performance matching frontier models on domain tasks, confirm distillation costs under $100, measure accuracy by task type. **Priority**: White's January-February 2025 posts documented the reasoning revolution; this phase capitalizes on small specialized models beating large generalists.

### Phase 3: Integration (Months 7-9) - Multi-agent coordination

**Build**: Deploy parallel specialized agents (literature, analysis, hypothesis generation), implement shared world model for agent coordination, create iterative cycle manager, add provenance tracking. **Validate**: Run multi-cycle investigations (10+ cycles), measure coherence over extended context, verify provenance completeness, evaluate against human expert performance. **Priority**: This builds toward Kosmos's core innovation - parallel agents with persistent shared memory enabling multi-cycle coherence.

### Phase 4: Scaling depth (Months 10-12) - Full discovery cycles

**Build**: Extend to 20+ cycle runs, add synthesis agent traversing world model to generate reports, implement quest-based steering, create external validation workflows. **Validate**: Independent expert evaluation of generated reports (target: 75%+ accuracy), measure time-equivalency vs. human researchers, attempt reproduction of known findings. **Priority**: This achieves Kosmos's demonstrated capability - 6-month equivalent investigations with ~80% accuracy.

### Phase 5: Production hardening (Month 13+) - Reliability and autonomy

**Build**: Improve synthesis accuracy (currently weakest at 57.9%), add spurious correlation detection to prevent "rabbit hole" descents, implement stopping criteria for unproductive directions, expand domain coverage beyond initial focus. **Validate**: Measure novel discovery rate, external peer review submissions, cost-per-validated-discovery at scale. **Priority**: Move from AI assistant to approaching true AI scientist definition.

### Critical path dependencies

Literature retrieval must precede hypothesis generation (need to check what's known). Small model specialization should precede multi-agent integration (need performant components before coordination overhead). Structured world models enable multi-cycle coherence; prioritize this architectural shift before scaling depth. External validation must run continuously from Phase 3 onward - epistemic honesty demands knowing accuracy at each stage.

### Risk mitigation strategies

**Technical risks**: World model scaling with investigation depth. Mitigation: Design for pruning irrelevant branches, implement hierarchical summarization, test on progressively longer runs. **Accuracy risks**: Synthesis errors and spurious correlations. Mitigation: Maintain validation checkpoints, require higher evidence thresholds for cross-domain synthesis, implement human-in-loop for major claims. **Economic risks**: API costs exceeding value. Mitigation: Aggressive model selection optimization, cached retrieval, progressive depth with early stopping.

## Cross-reference with Kosmos paper architecture

White's blog posts illuminate design choices that the formal Kosmos paper necessarily abbreviates. The paper presents structured world models enabling "coherent pursuit over 200 agent rollouts" and reports 79.4% accuracy with 6-month time equivalency. The blog reveals *why* these architectural choices were made:

**Paper states**: "Core innovation is structured world models." **Blog reveals**: This emerged from confronting coherence problem in earlier systems; context windows couldn't maintain focus beyond limited actions; structure enables querying early findings after millions of tokens; White specifically designed this component.

**Paper states**: "Two parallel agents - data analysis and literature search." **Blog reveals**: Specialization validated by reasoning experiments showing small domain-specific models exceed large generalists; parallel execution emerged from understanding that sequential chaining loses context; shared world model enables cross-pollination between analysis patterns and literature context.

**Paper states**: "79.4% accuracy (data analysis 85.5%, literature 82.1%, synthesis 57.9%)." **Blog reveals**: Synthesis weakness known limitation; White's epistemic honesty throughout blog prepares for this; synthesis requires combining evidence across domains - hardest capability; targeted area for improvement.

**Paper states**: "6-month time equivalency per 20-cycle run." **Blog reveals**: This "shocked us" per White; validated through beta users and reproduction studies; scales roughly 0.3 months per cycle; longer runs risk "rabbit holes" with spurious correlations.

**Paper omits**: Training recipes, cost models, reasoning architecture evolution. **Blog provides**: Three distinct training recipes (cold start, distillation, hybrid), complete cost breakdown ($75 for discovery analysis, $200 per Kosmos run, under $100 for distillation), evolution from complex MCTS approaches to simple generate-sample-vote strategies.

The blog reveals failed approaches informing better design - MCTS proved inefficient, Q-value prediction underperformed simple trajectory fine-tuning, human-legible reasoning isn't necessary for accuracy. The paper presents the working architecture; the blog documents the exploration leading there.

## Gaps and open questions the blog doesn't address

Despite comprehensive coverage, several critical questions remain unanswered:

### Technical gaps

**World model pruning strategies**: How to prevent world models from growing unboundedly over extended investigations? No discussion of hierarchical summarization, relevance decay, or selective forgetting. **Database schema specifics**: What exact entity types, relationship types, and property structures work best? Blog describes "entities and relationships" but not concrete implementations. **Concurrent investigation handling**: Can a single world model support multiple simultaneous quests, or does each require isolated instances? Implications for scaling.

### Methodological gaps

**Synthesis improvement pathways**: Synthesis accuracy (57.9%) is lowest, but no concrete proposals for improvement. Is this fundamental limitation of current LLMs, or architectural problem? Would different world model structures help? **Stopping criteria**: When should long runs terminate to avoid spurious correlations? How to detect "rabbit holes" before wasting compute? **Confidence calibration**: How to quantify and communicate uncertainty in world model contents? What makes a relationship "certain" vs. "speculative"?

### Economic gaps

**Scaling costs**: White provides single-run costs ($200 for Kosmos), but how do costs scale to 100-1000 simultaneous investigations? Are there economies of scale for shared infrastructure? **Marginal costs**: What's the cost to add one more cycle to an investigation? To expand world model with one more entity type? **Human time costs**: External validation requires expert time - how does this scale?

### Domain generalization gaps

**Beyond math and biology**: Reasoning model success concentrated in mathematics; world model demonstrations focus on biology/chemistry. How do these architectures generalize to physics, social sciences, humanities? **Non-verifiable domains**: White notes "you need a verifiable problem" but many scientific questions lack immediate ground truth (long-term ecological changes, social interventions). How to handle uncertainty? **Multimodal integration**: No discussion of incorporating images, videos, experimental sensor data into world models. Scientific discovery increasingly multimodal.

### Philosophical gaps

**Discovery novelty verification**: How to definitively determine if a finding is truly novel vs. buried in literature? Crow literature search helps but isn't perfect. **Societal value alignment**: "Quests" should reflect societal goals, but who decides? How to prevent optimization for publishable novelty over meaningful impact? **Reproducibility and validation**: What percentage of AI-generated discoveries will replicate in physical experiments? What's acceptable failure rate?

## Questions for the FutureHouse team

Based on blog analysis, priority questions for Andrew White and collaborators:

**Architecture decisions**: (1) What specific graph database schema and query patterns work best for biological world models? (2) How do you decide when to prune vs. persist information in extended runs? (3) What percentage of Kosmos's compute budget goes to world model queries vs. agent reasoning?

**Reasoning and synthesis**: (4) Why is synthesis accuracy (57.9%) so much lower than individual analysis (85.5%)? (5) What approaches are you exploring to improve cross-domain synthesis? (6) Can you quantify when reasoning vs. retrieval is the bottleneck?

**Scaling and economics**: (7) What's the marginal cost per additional cycle in Kosmos runs? (8) How do costs scale for supporting 100 concurrent users vs. 10? (9) At what point do world model storage/query costs exceed agent compute costs?

**Generalization**: (10) The infinite_discoveries post (d2-001, May 2025) wasn't accessible - what were the key insights from that post about scaling discovery processes? (11) How do you see world models generalizing beyond biology to physics, social sciences, engineering?

**Validation and trust**: (12) What validation strategies are you developing to improve on the 79.4% baseline? (13) How should the scientific community update its peer review processes for AI-generated discoveries? (14) What's your vision for reproducibility standards for AI scientist systems?

These questions target the boundaries of what the blog reveals, focusing on practical implementation details, known weaknesses, and paths toward the rigorous "AI Scientist" definition White articulated but admits no system yet achieves.

---

## Conclusion: From principles to production

Andrew White's blog posts document a remarkable year-long evolution from foundational economics to production AI scientist systems. The key architectural insight - **structured world models enabling persistent, queryable memory** - emerged from direct confrontation with the coherence problem limiting all previous attempts. Combined with reasoning breakthroughs showing small specialized models can exceed frontier generalists, and validated by Kosmos achieving 6-month research equivalents with ~80% accuracy, these principles provide a concrete roadmap for world model design.

The path forward is clear but demanding: build structured graph databases for entity-relationship storage, deploy parallel specialized agents communicating through shared world models, iterate through 20+ cycles with progressive refinement, maintain complete provenance for scientific credibility, validate ruthlessly with external experts, and remain epistemically honest about limitations. White's October 2025 definition sets an uncompromising standard - autonomous completion of the full discovery cycle from quest to peer-reviewed paper - that even Kosmos doesn't yet meet.

This represents not an endpoint but an inflection point. The architectural patterns are proven, the economic justification is overwhelming ($100M human discoveries vs. sub-$1,000 AI investigations), and the technical components exist. What remains is scaling depth while maintaining accuracy, improving synthesis capabilities, and expanding beyond mathematics and biology to general scientific discovery. White's blog provides both the inspiration and the instruction manual for that journey.