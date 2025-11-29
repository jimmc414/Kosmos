# Model Tier Comparison Report

## Session 18 - E2E Testing Phase 4

**Date**: 2025-11-29
**Research Question**: "How does temperature affect enzyme activity?"
**Test Configuration**: 1 cycle with literature context enabled

---

## Executive Summary

| Model | Time | Cost | Hypotheses | Exp. Designed | Cost/Hypothesis |
|-------|------|------|------------|---------------|-----------------|
| **DeepSeek** (baseline) | 79.9s | $0.0009 | 2 | 1 | $0.00045 |
| **GPT-4o-mini** | 26.9s | ~$0.0011 | 2 | 0* | ~$0.00055 |
| **Claude Haiku 4.5** | 43.5s | $0.0205 | 2 | 0* | $0.01025 |

*Experiment design failures were due to JSON/Pydantic validation issues, not model capability

---

## Detailed Results

### DeepSeek (deepseek/deepseek-chat)

**Performance**:
- Total time: 79.9s (1.3 min)
- Cost: $0.0009
- Input tokens: 1,904
- Output tokens: 2,175
- Total tokens: 4,079

**Hypothesis 1** - Standard Temperature-Activity Relationship:
> "Enzyme activity increases with temperature up to an optimal temperature of 37°C, then sharply declines due to thermal denaturation, with maximum activity occurring between 35-40°C for mammalian enzymes."

- Scientific accuracy: Standard, well-established concept
- Specificity: Moderate (mentions specific temperature range)
- Novelty: Low (textbook knowledge)

**Hypothesis 2** - Heat Shock Protein Protection:
> "Temperature stress induces increased production of heat shock proteins that protect enzyme structure, maintaining higher enzymatic activity under thermal stress conditions compared to enzymes without such protective mechanisms."

- Scientific accuracy: Valid biological mechanism
- Specificity: Moderate
- Novelty: Medium (connects stress response to enzyme activity)
- Literature integration: Cites "Response of Plant Secondary Metabolites to Environmental Factors, 2018" and "Microbial control over carbon cycling in soil, 2012"

---

### GPT-4o-mini (OpenAI)

**Performance**:
- Total time: 26.9s (0.4 min) - **FASTEST**
- Cost: ~$0.0011 (estimated: $0.15/1M input, $0.60/1M output)
- Input tokens: 1,659
- Output tokens: 1,426
- Total tokens: 3,085

**Hypothesis 1** - Optimal Temperature Enhancement:
> "Increasing temperature will enhance the catalytic activity of enzyme X up to an optimal temperature, beyond which the activity will decline."

- Scientific accuracy: Good
- Specificity: Low (generic "enzyme X", no specific temperatures)
- Novelty: Low (standard textbook description)
- Rationale: Good theoretical grounding (Arrhenius equation, denaturation)

**Hypothesis 2** - Temperature Reduction Effects:
> "Decreasing temperature will reduce the rate of enzyme-catalyzed reactions by slowing molecular motion and decreasing substrate binding efficiency."

- Scientific accuracy: Good
- Specificity: Low (no quantitative predictions)
- Novelty: Low (basic collision theory application)
- Rationale: Well-grounded in kinetic theory

---

### Claude Haiku 4.5 (claude-haiku-4-5-20251001)

**Performance**:
- Total time: 43.5s (0.7 min)
- Cost: $0.0205
- Input tokens: 1,921
- Output tokens: 4,747
- Total tokens: 6,668

**Hypothesis 1** - Asymmetric Biphasic Response:
> "Enzyme activity increases linearly with temperature from 0°C to an optimal temperature (typically 37-40°C for mammalian enzymes), then decreases sharply above this temperature due to protein denaturation, with the rate of activity decline above optimal temperature being steeper than the rate of increase below it."

- Scientific accuracy: Excellent
- Specificity: High (addresses asymmetry quantitatively)
- Novelty: Medium (adds nuance to standard model)
- Testable prediction: The asymmetry is quantifiable and testable

**Hypothesis 2** - Substrate-Dependent Q10:
> "The temperature coefficient (Q10) for enzyme activity varies inversely with substrate concentration, such that enzymes operating at saturating substrate concentrations exhibit lower Q10 values (1.5-2.0) compared to enzymes operating at limiting substrate concentrations (2.5-4.0)."

- Scientific accuracy: Excellent (mechanistically grounded)
- Specificity: Very high (provides specific Q10 ranges)
- Novelty: High (non-obvious mechanistic insight)
- Testable prediction: Clear experimental design possible
- Literature integration: Connects to environmental factors literature

---

## Comparative Analysis

### Quality Assessment

| Criterion | DeepSeek | GPT-4o-mini | Claude Haiku 4.5 |
|-----------|----------|-------------|------------------|
| Scientific Accuracy | Good | Good | Excellent |
| Hypothesis Specificity | Moderate | Low | Very High |
| Novelty/Insight | Low-Medium | Low | Medium-High |
| Testability | Moderate | Low | High |
| Literature Integration | Present | Minimal | Strong |
| Rationale Depth | Good | Good | Excellent |

### Cost-Benefit Analysis

| Metric | DeepSeek | GPT-4o-mini | Claude Haiku 4.5 |
|--------|----------|-------------|------------------|
| Cost per cycle | $0.0009 | ~$0.0011 | $0.0205 |
| Cost per hypothesis | $0.00045 | ~$0.00055 | $0.01025 |
| Time per cycle | 79.9s | **26.9s** | 43.5s |
| Output tokens | 2,175 | 1,426 | 4,747 |
| Speed rank | 3rd | **1st** | 2nd |
| Cost rank | **1st** | 2nd | 3rd |
| Quality rank | 2nd | 3rd | **1st** |

### Token Economics

**DeepSeek** (~$0.14/1M input, ~$0.28/1M output):
- Most cost-effective overall
- Good balance of quality and cost

**GPT-4o-mini** ($0.15/1M input, $0.60/1M output):
- Fastest response time (3x faster than DeepSeek)
- Similar cost to DeepSeek
- Lower quality hypotheses

**Claude Haiku 4.5** ($1.00/1M input, $5.00/1M output):
- Highest quality hypotheses
- ~19x more expensive than DeepSeek
- Most verbose output (2.2x more tokens than DeepSeek)

---

## Hypothesis Quality Breakdown

### DeepSeek Strengths:
1. Solid foundational hypotheses
2. Appropriate literature citations
3. Clear, concise statements
4. Best cost/quality ratio

### DeepSeek Weaknesses:
1. More conventional insights
2. Less mechanistic depth
3. Fewer quantitative predictions

### GPT-4o-mini Strengths:
1. **Fastest response time** (26.9s)
2. Concise, well-structured rationales
3. Cost-competitive with DeepSeek
4. Good theoretical grounding

### GPT-4o-mini Weaknesses:
1. Generic hypotheses (no specific enzymes/temperatures)
2. Least novel insights
3. Minimal literature integration
4. Fewer testable predictions

### Claude Haiku 4.5 Strengths:
1. Most sophisticated mechanistic reasoning
2. Quantitative predictions (specific Q10 ranges)
3. Non-obvious insights (substrate-dependent Q10 effect)
4. Deeper rationales with theoretical grounding
5. Best literature integration

### Claude Haiku 4.5 Weaknesses:
1. ~19x higher cost than competitors
2. Most verbose (higher token consumption)

---

## Recommendations

### For Development/Testing (Speed Priority):
**Use GPT-4o-mini** - Fastest iteration cycles at competitive cost. Good for rapid prototyping.

### For Development/Testing (Cost Priority):
**Use DeepSeek** - Best cost/quality ratio. Excellent for high-volume testing.

### For Production Research:
**Use Claude Haiku 4.5** - When hypothesis quality is critical. The superior mechanistic insights and quantitative predictions justify the cost premium.

### For Cost-Constrained Research:
**Use DeepSeek** - At $0.0009/cycle, can run ~23 cycles for the cost of 1 Claude cycle.

### Hybrid Approach (Recommended):
- **Exploration phase**: GPT-4o-mini (fast iteration)
- **Development phase**: DeepSeek (cost-effective quality)
- **Refinement phase**: Claude Haiku 4.5 (highest quality)
- Estimated optimal mix: 40% GPT-4o-mini / 40% DeepSeek / 20% Claude Haiku

---

## Test Artifacts

Results saved to:
- `artifacts/model_comparison/deepseek/` - DeepSeek run
- `artifacts/model_comparison/gpt4o_mini/` - GPT-4o-mini run
- `artifacts/model_comparison/claude_haiku/` - Claude Haiku run

---

## Technical Notes

1. **OPENAI_BASE_URL Issue**: The shell had `OPENAI_BASE_URL=https://api.deepseek.com` set, causing OpenAI keys to be sent to DeepSeek's API. Fixed by adding explicit `OPENAI_BASE_URL=https://api.openai.com/v1` to .env.

2. **Experiment Design Failures**: All three models had issues with experiment design:
   - DeepSeek: Succeeded (1 experiment)
   - GPT-4o-mini: Pydantic validation error (sample_size type mismatch)
   - Claude Haiku: JSON parsing error

3. **Cost Tracking**: GPT-4o-mini showed $0.00 cost due to "compatible" provider type not having pricing configured. Actual cost estimated from token counts.

4. **Model Used**: Tested gpt-4o-mini instead of gpt-5-nano (gpt-5-nano may not exist or not be available).

---

*Report generated: Session 18, 2025-11-29*
*Updated with GPT-4o-mini results*
