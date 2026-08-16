# domestic-violence-risk
Probabilistic Risk for Domestic Violence Femicide: A Bayesian Network Model and Time-Based Adaptation of the Danger Assessment Instrument

A Bayesian Network model and time-based adaptation of the Danger Assessment (DA) instrument — a 19-item, clinically-validated screening tool used by police and advocates to gauge lethality risk after a domestic violence incident.

**Author:** Dr Alex Bevan
**Notebook:** `main.ipynb`

## DISCLAIMER: THIS IS NOT A RISK ASSESSMENT TOOL THAT SHOULD BE USED. It is for research purposes only and is part of a Masters Degree in Applied Data Science.

## Overview

The Danger Assessment (Campbell, 1986; Campbell, Webster & Glass, 2009) is a weighted 19-item yes/no questionnaire — "Does he own a gun?", "Has he threatened suicide?" — that sums to a score mapped onto four risk categories. It was built on a logistic regression fit to an 11-city case-control study (310 femicide/attempted-femicide cases vs. 324 controls); 90% of homicides in that study scored in the top two categories.

This project re-encodes the DA as a **Bayesian network**: each of the 19 risk factors is a node with a directed edge into a single `lethality` node, with the network's conditional probability table (CPT) derived via a **leaky noisy-OR** model. It then extends the DA with a **novel time/recency dimension** on two items (escalation, victim leaving), and compares the resulting expert-encoded model against a **data-driven model** learned from a synthetic dataset built from real femicide news coverage.

This is one component of a larger capstone proposing to optimize police patrol routing for domestic violence outcomes.

## Data

Real DV case data is extremely hard to access (IRB timelines, institutional gatekeeping), so this project simulates a dataset from **news stories on femicide and attempted femicide (2015–2026, USA, straight couples)**, gathered via an LLM research agent prompted to extract DA-item answers from case reporting. This introduces significant selection bias but was the most viable option available.

- Base dataset: 68 fatal + 18 attempted cases, averaging only 2.9/19 questions answered per case from news text.
- Sparse categories (coercive sex, drug use, pregnancy abuse, unemployment, "capable of killing," non-biological child) were imputed using published population base rates for abusive relationships.
- Fatal cases were split at the pre-fatal event to synthesize matched "survived" counterfactual cases, expanding the set to 154 rows.
- Files live in `Bayes DF/`, with `final_df_manual_cohab.csv` as the version used by the notebook.

## Key Assumptions

1. The DA is treated as current, though it hasn't been revised since 2009.
2. Same-sex violence is not represented — the source literature and news sample are heavily skewed toward heterosexual couples.
3. Risk factors are modeled as conditionally independent given lethality (standard noisy-OR assumption), which is known not to hold in reality.

## Methodology

### 1. Encoding the DA as a Bayesian Network
Each of the 19 DA items becomes a binary node (`gun ownership`, `nonfatal strangulation`, `controlling behavior`, etc.) with an edge into `lethality`.

### 2. Leaky Noisy-OR CPT with a Derived Prior
Rather than hand-specifying 2^19 CPT rows, item weights (from the DA's original clinical weighting scheme) are normalized to per-factor probabilities via noisy-OR, with a **leak probability** representing baseline lethality risk when no modeled factor is present. The leak is derived via Bayes' theorem:

P(femicide | abuse) = P(abuse | femicide) · P(femicide) / P(abuse)

using published estimates (UNODC regional femicide rates, Campbell et al. 2003 for P(abuse | femicide), WHO lifetime abuse prevalence), scaled from an annual to a ~40-year lifetime exposure window. The full 2^19-row CPT is materialized and saved to `lethality_cpd_full.csv`.

### 3. Encoding Time (novel contribution)
The original binary `escalation` and `victim left` items are replaced with 6-state **recency** nodes (`<2wks`, `<1m`, `<3m`, `<6m`, `<1yr`, `None`) with linearly decaying weight, so more recent escalation or leaving events carry more risk than the original binary encoding allowed.

### 4. Case Studies
Three hand-constructed evidence profiles are run through the network via variable elimination to sanity-check posterior `lethality` probabilities.

### 5. Data-Driven Model (Hill Climb + MLE)
A second Bayesian network structure is *learned* from the news-derived dataset using Hill Climb search (BIC scoring, `lethality` forbidden as a parent) with parameters fit via Maximum Likelihood Estimation. This model reflects how news media *narrates* DV rather than DV's true causal structure — strangulation, weapon threats, controlling behavior, and escalation emerge as the dominant "story" drivers.

### 6. Evaluation
Both models are evaluated on a held-out test split (grouped by original case, so synthetic near-duplicates don't leak across the split) using accuracy, precision/recall/F1, ROC-AUC, and Youden's-J threshold optimization.

## Key Findings

- The data-driven (Hill Climb) model essentially collapses to the majority-class baseline — the news-derived dataset is too small and sparse (avg. 2.9/19 items per case) to learn reliable lethality prediction, and it captures **media framing of DV**, not DV's actual dynamics.
- The expert/clinical model with time-modification consistently outperforms the data-driven model.
- Both models' predicted probabilities top out in the 60s%, well under the standard 0.5 classification threshold, so both default to predicting "not lethal" for nearly everything.
- Lowering the decision threshold to the empirically-optimal ~0.125 (Youden's J) substantially improves recall on lethal cases at the cost of a high false-positive rate — a real and unresolved tradeoff in a high-stakes screening context (better to over-flag than miss a lethal case, but at what cost to those flagged?).
- The time-recency contribution shows a modest positive effect, suggesting some real signal connects escalation/leaving timing to lethality even in this noisy proxy dataset.

## Repository Contents

| File | Description |
|---|---|
| `main.ipynb` | Main analysis notebook (latest version) | and html version
| `Bayes DF/final_df_manual_cohab.csv` | Primary dataset used for training/evaluating the data-driven model (154 rows, incl. synthetic counterfactuals) |

## Environment / Dependencies

Python environment named `bayes_ml`. Core libraries: `pandas`, `numpy`, `pgmpy` (Bayesian networks, Hill Climb, MLE, variable elimination), `pymc`, `arviz`, `causalgraphicalmodels`, `graphviz`, `networkx`, `scikit-learn` (`GroupShuffleSplit`, classification metrics, ROC-AUC), and `probml_utils` (installed from GitHub — `pip install git+https://github.com/probml/probml-utils.git --no-deps`). The setup cell in the notebook installs all non-standard dependencies automatically.

## Key References

- Campbell, J. C. (1986). Nursing assessment for risk of homicide with battered women. *Advances in Nursing Science*, 8(4), 36–51.
- Campbell, J. C., Webster, D. W., & Glass, N. (2009). The Danger Assessment: Validation of a Lethality Risk Assessment Instrument for Intimate Partner Femicide. *Journal of Interpersonal Violence*.
- Campbell, J. C. et al. Risk Factors for Femicide in Abusive Relationships: Results From a Multisite Case Control Study. *American Journal of Public Health*.
- WHO — Violence Against Women fact sheet (lifetime abuse prevalence).
- UNODC — regional femicide rate data.

## Limitations

This is a class capstone project, not a validated clinical or operational tool. The training dataset is small, selection-biased (news coverage only), and heavily imputed; the independence assumption underlying noisy-OR does not hold for real DV risk factors; and same-sex relationships are entirely unrepresented. Findings here should be read as a methodological exploration of encoding a clinical instrument as a probabilistic graphical model, not as a deployable risk-prediction system.

