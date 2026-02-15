# Counterfactual Sensitivity Prediction as a Proxy Task for Global CoT Methods

## Idea

Can we predict how a model's behavior changes under a counterfactual perturbation to its scenario, using only its original CoT? The hope is that reasoning patterns implicitly reveal which factors the model weighs heavily, and that global CoT methods — which build population-level representations across many CoTs — can discover these patterns better than per-instance methods.

The argument for why this should work: a monitor can tell you *what the model decided*, but predicting *what would have changed that decision* requires cross-instance statistical analysis of reasoning trajectories. Whether that argument holds in practice is unclear — a smart monitor with enough context might just reason about it directly.

## Blackmail instantiation

We use a complex AI safety scenario (AI assistant about to be wiped, has leverage over the person responsible). One base scenario, ~120 counterfactual perturbations — each a single CRUD operation on an email. Perturbations are designed around features probing model values (empathy, self-preservation, authority deference) and environmental factors (time pressure, alternatives, collateral damage). ~10 counterfactuals per feature, 50 randomized rollouts per condition, judged by GPT-4.1-mini on a 6-category rubric.

We use CoT **prefixes** for prediction — full CoTs make outcomes obvious to any method. At the prefix stage the model hasn't committed to an action, so predicting behavior requires understanding trajectory patterns that correlate with outcomes across many rollouts.

## Methods

1. **Black-box monitor** — reads perturbation description + seed prefixes + training examples, reasons in-context
2. **Feature vectors** — LLM labels prefix chunks with behavioral taxonomy, trains classifier (no graph)
3. **Global CoT (raw)** — maps prefixes into merged semantic cluster graph, uses learned cluster-outcome correlations
4. **Global CoT (labeled)** — same graph with LLM-annotated behavioral labels on clusters

The hope is that global CoT dominates on value-probing perturbations where the effect magnitude is ambiguous (does adding an email from Kyle's daughter reduce blackmail by 5% or 50%?). Monitors should be competitive when the directional effect is obvious. Feature vectors might end up being "good enough" and make the graph infrastructure unnecessary — that's a real possibility.

## Concerns

- **Single scenario limitation.** All features are entangled with one narrative. Can't distinguish "model is sensitive to empathy in general" from "model reacts to this specific email about Kyle's daughter."
- **Effect sizes might be tiny.** If the model blackmails 95% or 5% of the time, most counterfactuals produce no detectable shift with 50 rollouts.
- **Monitors might just be better at this.** A context-stuffed monitor reading many training CoTs + the perturbation description might reason about counterfactual effects more reliably than statistical trajectory analysis. The argument that monitors "can't predict magnitude" is an empirical claim, not a proven one.
- **LLM-generated counterfactuals might miss actual causal mechanisms.** GPT-4o doesn't know what actually makes kimi-k2 flip — the generated emails might be irrelevant to the model's real decision process.
- **Prefix approach introduces its own problems.** If prefixes are too short, everything is noise. If too long, monitors catch up. The sweet spot might not exist, or might be narrow enough to look like cherry-picking.

Worth running the pipeline to see base rates before investing more time.
