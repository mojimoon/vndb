# Ranking method experiments

This file records how the set of ranking methods and the composition of the
grand ranking (`borda_grand`, shown as "SciRanking") were chosen: which methods
existed, which were removed and why, and the experiments behind each decision.

All numbers come from the VNDB dump of **2026-10-06**: 7,945 titles with at
least 30 votes, 1.87M votes from 92,580 users (users VNDB flags as ignored are
excluded), and 4.89M title pairs with at least 5 common voters.
`research/grand_experiment.py` reproduces the grand-ranking study.

## Methods and terms

**Game.** Every pair of titles with at least 5 common voters is treated as a
game between them. Each method turns the set of games into one score per title.

**Game-score variables** (how a game is scored):

| Variable | Score of A vs. B |
|---|---|
| `prob` | number of common voters who rated A higher / B higher |
| `ari`, `geo` | arithmetic / geometric mean of the common voters' votes |
| `sp_ari`, `sp_geo` | arithmetic / geometric mean of the votes' *sample percentiles*, i.e. where the vote sits within that voter's own list, so a user who rates everything 8 still separates favourites |

**Algorithms.**

* PONet (partial-order network) methods work directly on the pair table:
  `po_total`, `po_percent`, `po_simple`, `po_weighted` (variants of net
  preference), `po_elo`, `po_entropy`, `po_bt` (Bradley–Terry).
* rankit algorithms are applied to each variable: Massey (least-squares rating
  differences), Colley (win/loss record adjusted for opponent strength), Keener
  (Perron eigenvector of the score matrix), Markov chains (`markov_rv`: weight
  flows to the winner in proportion to its win share; `markov_rdv`: to the
  normalised margin; `markov_sdv`: to the raw margin), offence–defence (`od`),
  and Difference (mean score margin).
* Borda merges add up the ranks a title gets from several methods:
  score = N × M − Σ ranks, with N titles and M methods (higher is better; the
  average rank over the merged methods is N − score / M). Methods that cannot
  rank a title count it as tied last.

**Metrics used below.**

| Metric | Meaning |
|---|---|
| τ (consensus) | Kendall τ between a ranking and the consensus, the median rank of each title over all 47 methods |
| ρ (popularity) | Spearman ρ between a ranking's scores and log(vote count). For reference, VNDB's Bayesian rating has ρ = 0.34 and the plain mean vote 0.30 |
| split-half reliability | the voters are split into two random halves of equal size, every method is computed on each half, and the two rankings are compared (Spearman) |
| reliability beyond popularity | the same, with log(vote count) partialled out of both halves: how consistently the method ranks titles *apart from* how popular they are |
| top-100 stability | how many titles the two halves' top 100 share |

## History of the method set

| Version | Methods | Change |
|---|---|---|
| legacy script (`research/legacy_main.py`) | 8 PONet + 40 rankit + 8 Borda | |
| v1 (pipeline rewrite) | 56 + VNDB reference | `po_vi` replaced by `po_bt`; bugs fixed (below) |
| v2 | same as v1 | |
| v3 | 39 + VNDB reference | 17 methods removed (below) |
| v4 | same as v3 | |
| v5 | 41 + VNDB reference | `difference_sp_ari`, `difference_sp_geo` added back; grand ranking changed to G2 (below) |

### v1: replaced and fixed

* `po_vi` (variational inference) took unseeded random gradient steps, so results
  changed from run to run, and its fancy-indexed updates silently dropped
  duplicate pairs. Replaced by a Bradley–Terry model fitted with the MM
  algorithm (`po_bt`), the model it approximated.
* `po_elo`: count-weighted updates diverged, and ratings were truncated to
  integers. It now plays one match per pair whose outcome is the preference
  share, over shuffled passes with decaying K.
* `po_entropy` subtracted one side of the entropy normalisation instead of
  adding it.
* The rankit Markov variants transformed the shared input frame in place, so
  every ranker after them saw altered scores.
* The pair table indexed items by rating order but iterated in id order, so
  about half of all comparisons landed in the lower triangle and were dropped
  on export.

### v3: removed after a method audit

Every method was compared with the consensus (τ) and with popularity (ρ):

| ρ with log votes | `prob` | `ari` | `geo` | `sp_ari` | `sp_geo` |
|---|---|---|---|---|---|
| massey | −0.04 | 0.38 | 0.36 | 0.40 | 0.40 |
| colley | 0.42 | 0.41 | 0.40 | 0.43 | 0.41 |
| keener | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| markov_rv | 0.76 | 0.77 | 0.76 | 0.77 | 0.77 |
| markov_rdv | 0.71 | 0.75 | 0.74 | 0.75 | 0.72 |
| markov_sdv | 0.77 | 0.73 | 0.72 | 0.71 | 0.68 |
| od | 0.42 | 0.47 | 0.45 | 0.46 | 0.45 |
| difference | −0.09 | 0.26 | 0.26 | 0.25 | 0.23 |

| τ with consensus | `prob` | `ari` | `geo` | `sp_ari` | `sp_geo` |
|---|---|---|---|---|---|
| massey | 0.41 | 0.75 | 0.74 | 0.75 | 0.75 |
| colley | 0.73 | 0.74 | 0.73 | 0.73 | 0.74 |
| keener | 0.57 | 0.40 | 0.40 | 0.42 | 0.42 |
| markov_rv | 0.63 | 0.42 | 0.43 | 0.50 | 0.53 |
| markov_rdv | 0.79 | 0.78 | 0.78 | 0.78 | 0.77 |
| markov_sdv | 0.76 | 0.80 | 0.80 | 0.81 | 0.82 |
| od | 0.77 | 0.80 | 0.78 | 0.81 | 0.81 |
| difference | 0.35 | 0.62 | 0.61 | 0.58 | 0.57 |

PONet methods: `po_total` τ 0.62 / ρ 0.20, `po_percent` 0.83 / 0.54,
`po_simple` 0.82 / 0.54, `po_weighted` 0.76 / 0.42, `po_elo` 0.75 / 0.40,
`po_entropy` 0.79 / 0.51, `po_bt` 0.73 / 0.39.

Removed (17):

| Removed | Reason |
|---|---|
| `po_rw` (random walk) | τ −0.04: uncorrelated, and with some methods negatively correlated (−0.25..0.19) |
| `keener_*` (5) | τ 0.40–0.57 and ρ ≈ 0.75: mostly a popularity contest |
| `markov_rv_*` (5) | τ 0.42–0.63 and ρ ≈ 0.77: same |
| `difference_*` (5) | τ 0.35–0.62; its top lists were the most-voted titles |
| `massey_prob` | τ 0.41: raw preference counts let pairs with huge audiences dominate the least-squares fit |

`po_total` was kept but taken off the featured list: bigger audiences give
bigger margins, so it favours popular titles at the top.

Removing these also shrank the merges: `borda_po` lost `po_rw`, each
`borda_<variable>` went from 8 algorithms to 5 (4 for `prob`), and
`borda_grand` from 18 inputs to 14.

## The grand ranking

### Compositions

* **Legacy, v1, v2 (18 inputs):** Massey, Colley, Markov (share margin), Markov
  (score margin), offence–defence and Difference, each on `prob`, `sp_ari` and
  `sp_geo`.
* **v3, v4 (14 inputs):** the same without the removed `difference_*` and
  `massey_prob`.
* **v5, "G2" (9 inputs, current):** `massey_sp_ari`, `massey_sp_geo`,
  `colley_prob`, `colley_sp_ari`, `colley_sp_geo`, `difference_sp_ari`,
  `difference_sp_geo`, `po_bt`, `po_elo` (`methods.GRAND_INPUTS`). The two
  Difference inputs were added back as regular methods in v5; the other three
  Difference variables stay removed. The per-variable merges
  (`borda_sp_ari`, `borda_sp_geo`) still use only the five v3 algorithms.

### Problem

The v3 grand ranking looked more popularity-driven than v2, and it was:
ρ rose from 0.50 to 0.59. The cause is the composition. 6 of the 14 v3 inputs are
the margin-based Markov variants (ρ ≈ 0.72); in v2 the three `difference_*`
inputs (ρ ≈ 0.25) diluted them. At the very top the effect went the other way
(v3's top 100 has a median of 1,563 votes against v2's 2,256), because
Difference favours well-established titles at the top while barely tracking
popularity overall.

The question was whether a semantically meaningful combination could keep
agreement while lowering ρ.

### Why τ against the consensus cannot be the target

About 20 of the 47 methods behind the consensus are popularity-driven (Keener,
both win-share and margin Markov, ρ 0.68–0.77), so the consensus itself leans
towards popularity. In an exhaustive search over every subset of the 6 grand
algorithms × every subset of the 5 variables (1,953 combinations), every
combination with τ ≥ 0.85 and ρ ≤ 0.5 needed Keener or win-share Markov, the
methods removed in v3 for being popularity contests.

### Split-half reliability

Split-half reliability needs no reference ranking: a combination that ranks
titles the same way from two independent sets of voters is measuring something
real. Plain split-half reliability turned out to reward popularity as well.
Vote counts are very consistent between halves, so Keener and win-share Markov
were the most "reliable" single methods (≈ 0.90). The deciding metric is
therefore reliability **beyond popularity** (log vote count partialled out):

| Reliability beyond popularity | `prob` | `ari` | `geo` | `sp_ari` | `sp_geo` |
|---|---|---|---|---|---|
| massey | 0.83 | 0.85 | 0.83 | 0.84 | 0.83 |
| colley | 0.82 | 0.80 | 0.79 | 0.80 | 0.80 |
| keener | 0.69 | 0.66 | 0.66 | 0.66 | 0.66 |
| markov_rv | 0.68 | 0.64 | 0.64 | 0.65 | 0.67 |
| markov_rdv | 0.68 | 0.63 | 0.64 | 0.65 | 0.67 |
| markov_sdv | 0.64 | 0.66 | 0.66 | 0.67 | 0.69 |
| od | 0.72 | 0.66 | 0.66 | 0.69 | 0.70 |
| difference | 0.72 | 0.72 | 0.70 | 0.72 | 0.70 |

PONet: `po_bt` 0.85, `po_elo` 0.83, `po_total` 0.76, `po_percent` 0.69,
`po_weighted` 0.69, `po_simple` 0.65, `po_entropy` 0.64.

The margin-based Markov variants, 6 of the 14 v3 grand inputs, were the least
reliable beyond popularity *and* among the most popularity-driven. Massey,
Colley, Bradley–Terry and Elo were the most reliable.

### Candidates

All searched over rankit subsets and PONet subsets with the metrics above;
these are the semantically defined ones worth keeping:

| Combination | Inputs | Reliability | Beyond popularity ↑ | ρ ↓ | τ (consensus) | Top-100 stability ↑ | Top 100 with < 100 votes |
|---|---|---|---|---|---|---|---|
| legacy / v1 / v2 grand | 18 | 0.849 | 0.787 | 0.497 | 0.865 | 85 | 0 |
| v3 / v4 grand | 14 | 0.868 | 0.771 | 0.585 | 0.904 | 83 | 2 |
| A: Massey + Colley × 3 variables | 5 | 0.859 | 0.830 | 0.411 | 0.746 | 65 | 38 |
| C: A + Bradley–Terry + Elo | 7 | 0.864 | 0.838 | 0.407 | 0.746 | 69 | 34 |
| F: Massey + Colley + OD × 3 | 8 | 0.858 | 0.810 | 0.426 | 0.774 | 70 | 30 |
| G: C + Difference × 3 | 10 | 0.822 | 0.807 | 0.338 | 0.713 | 83 | 0 |
| **G2: G without `difference_prob`** | **9** | 0.838 | **0.818** | **0.380** | 0.737 | 81 | **0** |

The pure strength-model blends (A, C, F) are the most reliable beyond
popularity but fail in practice: about a third of their top 100 are titles with
fewer than 100 votes (C's #1 had 87 votes, #3 had 34), and that top list is the
least stable between halves. Difference on percentile margins works as an
evidence anchor. It restores a stable top with no thinly voted titles and
lowers ρ further.

G2 was chosen over G because it drops the second count-margin outlier
(`difference_prob`, τ 0.35 and ρ −0.09, the same pathology as `massey_prob`),
which makes the rule simple: raw preference counts are used only by Colley,
which only looks at wins and losses anyway. G2 is also slightly more reliable.
G lowers ρ further (0.34) if popularity should be pushed down even more.

Compared with v3/v4, G2 is more reliable beyond popularity (0.818 vs. 0.771)
and much less popularity-driven (ρ 0.38 vs. 0.59, close to VNDB's own 0.34),
with an equally stable top list. Kendall τ between G2 and v2 is 0.86.

G2 top 15 (votes, v4 rank): Rance X -Kessen- (1,240, #1), Ore-tachi ni Tsubasa
wa Nai (645, #3), Utawarerumono: Futari no Hakuoro (2,969, #2), Tsukihime -A
piece of blue glass moon- (3,000, #4), BALDR SKY Dive2 (2,732, #5), Sakura no
Uta (2,577, #6), Dai Gyakuten Saiban 2 (1,686, #9), Kitto, Sumiwataru Asairo
Yori mo, (568, #10), WHITE ALBUM2 (4,615, #8), Saihate no Ima (297, #7),
Sakura no Toki (1,101, #11), Higurashi no Naku Koro ni Kai (8,037, #14),
Kajiri Kamui Kagura (404, #12), Gyakuten Kenji 2 (1,949, #22), Mahoutsukai no
Yoru (5,573, #15).

### Caveats

* One dump and one random split of the voters. The differences between the
  candidates are clear but not large; rerun `research/grand_experiment.py` on a
  later dump (it splits users by a fixed hash) before changing the composition
  again.
* Some correlation with popularity is legitimate: better games attract more
  players. The goal was to stop the ranking from *rewarding* popularity beyond
  that, not to reach ρ = 0.
* Reliability beyond popularity partials out log(vote count) linearly; titles
  near the 30-vote threshold are noisy under every method.
