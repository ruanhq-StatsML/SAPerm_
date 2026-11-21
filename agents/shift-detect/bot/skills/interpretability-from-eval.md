---
description: Explain model behavior from the evaluation lens (factor attribution)
---

# Interpretability from evaluation

This agent does **not** start from saliency maps. It explains behavior by
locating which **eval axes** failed and mapping them to operational factors:

| Weak axis | Likely factor |
| --- | --- |
| safety / refusal_quality | safety_policy |
| groundedness | rag_retrieval / context |
| instruction_following | prompt_design |
| multimodal_alignment | modality_encoder |
| reasoning / factuality | model_capability |
| consistency | decoding |

Use `explain_llm_behavior` then, if severe, `promote_to_dataset`.

Always return: primary factor, secondary factors, evidence quotes from the
scorer, and a concrete fix. That is the interpretability deliverable.
