# Clienteling

Which clients each advisor should contact this quarter, judged by the sales a contact adds rather than by how many contacts are made, and which pieces to show them.

```bash
python scripts/build_clienteling.py       # pilot, uplift models over 3 pilot draws, policies, power analysis (about 8 minutes)
python -m scripts.pilot_size_value        # does a bigger pilot help? (about 1 minute)
python scripts/build_recommendations.py   # product picks on real purchases (about 15 minutes)
```

Outputs: `reports/clienteling.json`, `reports/clienteling_policies.csv`, `reports/recommendations.json`, and the action list `lux.clienteling_actions`.

## What is real and what is simulated

No public data records luxury advisor outreach and its effect, so the project separates the two parts:

| Part | Source |
|---|---|
| Each client's spend and visits in the 90 days after a cut-off, with no contact | **Real** (Visuelle) |
| Clients, advisors, consent, euro values | Calibrated luxury layer |
| The effect of one advisor contact | **Simulated** from a documented rule (`configs/clienteling.yml`) |
| Product purchases used to judge product picks | **Real** |

The contact-effect rule: a contact raises the chance of an extra visit within 90 days. The effect is largest for engaged clients who are starting to drift (about 13 weeks since their last visit), small for clients who would come back anyway or have gone cold, and **negative** for top clients contacted right after a purchase ("do not disturb"). Each client also has a fixed, unobservable responsiveness. Because the true effect is known, every method can be judged against it.

## Set-up

1. **Pilot (30 June 2018):** half of the 198,175 eligible clients (consented, seen in the last two years) are contacted at random.
2. **Choosing the uplift model:** six learners are trained on 80% of the pilot and compared by Qini area on the other 20%, using observed outcomes only, as a real team would have to.
3. **Deployment (30 June 2019, same season):** every policy gets the same capacity, 40 contacts per advisor per quarter across 2,732 advisors, and is scored by the true incremental euros of the clients it contacts.

## Choosing an uplift model from the pilot alone

Eight learners were compared on the pilot (simplest first): logistic T-learners on core or all features, gradient-boosting T-learners, a transformed-outcome model, an X-learner on spend, a regularised logistic S-learner, and a bagged logistic T-learner.

| Learner | Qini on pilot holdout [95% CI] | Ranking stability across bootstrap refits | Correlation with the true effect |
|---|---|---:|---:|
| t_logistic_core | 16.8 [-19.7, 68.0] | 0.636 | 0.318 |
| t_logistic_full | 37.9 [6.3, 88.3] | 0.647 | 0.34 |
| t_gbm_full | 45.2 [2.7, 106.7] | 0.406 | 0.093 |
| t_gbm_full_large_leaves | 53.3 [21.4, 94.8] | 0.43 | 0.024 |
| transformed_outcome_core | 53.3 [11.5, 102.7] | 0.402 | 0.023 |
| x_learner_spend | -14.5 [-57.9, 36.8] | – | 0.029 |
| s_logistic_core | 43.4 [4.5, 85.6] | 0.644 | 0.357 |
| bagged_t_logistic_core | 28.6 [-8.8, 74.0] | 0.553 | 0.392 |

The average contact effect is small (1.7 points on the chance of a visit), so the Qini intervals all overlap: seven of eight learners are statistically indistinguishable from the best, and the highest Qini belongs to models that barely recover the true effect. Two pilot-only checks are therefore used, both fixed before looking at deployment results:

1. **Choose:** among learners indistinguishable from the best Qini, the one whose client ranking is most stable across bootstrap refits of the pilot. Chosen: `t_logistic_full` (stability 0.647).
2. **Trust check:** uplift targeting is only trusted if that stability is at least 0.5; otherwise the pilot is too noisy and the house keeps its current targeting.

## How much does the result depend on one pilot?

The pilot was re-randomised three times and the whole choose-and-deploy process repeated (share of the best achievable incremental euros at 40 contacts per advisor):

| Pilot draw | Learner chosen | Stability | Trusted? | Uplift policy | With the trust check |
|---|---|---:|---|---:|---:|
| 0 | t_logistic_full | 0.647 | yes | 81.7% | 81.7% |
| 1 | t_logistic_full | 0.546 | yes | 79.6% | 79.6% |
| 2 | t_gbm_full_large_leaves | 0.405 | **no** | 57.4% | 87.4% |
| **Mean** | | | | **72.9%** | **82.9%** |
| "Call your best clients" | | | | 87.4% | 87.4% |

- The trust check works: it flags the bad pilot draw and lifts the worst case from 57.4% to 79.6%.
- **But a trusted uplift model still did not beat calling the best clients** in two of two trusted draws. Stability is necessary, not sufficient.
- In hindsight the core-feature variants (bagged 89.3%, S-learner 89.4%) did beat the status quo on the first draw, but the pilot could not single them out: choosing them now would be tuning to the simulation's known truth.

**Is a bigger pilot the fix?** Re-running with 25%, 50%, 100% and 200% of a quarter's eligible clients (the last by pooling March and June 2018), three draws each, gave no reliable improvement: run-to-run variation swamps the effect of size (`reports/pilot_size_value.csv`). More data per client (a longer horizon, or a cleaner outcome such as appointments booked) would help more than more clients.

## Recommendation: champion and challenger

Calling each advisor's most valuable clients stays the deployed policy (the **champion**); in euros it already captures 87.4% of what is achievable, because a top client's typical visit is worth many times a mid-tier client's. The trusted uplift policy runs as a randomised **challenger**: advisors are split between the two lists and the challenger replaces the champion only if the live test shows more visits. `lux.clienteling_actions` carries both actions for every client.

Results on the first pilot draw:

| Policy (104,869 contacts) | True incremental € | € per contact | Share of best possible |
|---|---:|---:|---:|
| Random | €27.2M | €260 | 37.3% |
| Highest predicted spend if contacted | €63.3M | €603 | 86.7% |
| Churn risk × value | €63.8M | €608 | 87.3% |
| **Call your best clients (champion)** | €63.8M | €609 | 87.4% |
| Uplift challenger (chosen learner) | €59.7M | €569 | 81.7% |
| Bagged logistic T-learner, core features (hindsight) | €65.2M | €622 | 89.3% |
| Oracle (true effect known) | €73.0M | €698 | 100.0% |

| Contacts per advisor per quarter | Champion | Challenger |
|---|---:|---:|
| 10 | 79.3% | 69.6% |
| 20 | 84.8% | 76.4% |
| 40 | 87.4% | 81.7% |
| 80 | 89.6% | 88.0% |

13–17% of contacts under any practical policy slightly reduce sales, mostly through unobservable client responsiveness; only the observable "do not disturb" pattern can be avoided. The uplift scores are kept as a `predicted_harmful` flag advisors can see.

## Experiment design for the challenger

Among the clients the challenger would contact:

| Success metric | Effect per contact | Spread | Holdout needed (client randomised, with CUPED) | Holdout needed (whole advisors, with CUPED) |
|---|---:|---:|---:|---:|
| Euros spent in 90 days | €569 | SD €47,039 | Not detectable with up to a 50% holdout | Not detectable |
| **Visited in 90 days** | **+1.8 points** | SD 0.43 | **5%** | **17%** |

Measure on visit rate, with euros as a secondary readout; use last-12-month behaviour as a CUPED covariate. Randomising whole advisors is operationally cleaner but needs a larger share because clients of the same advisor behave alike (design effect 3.42).

## Action list

`lux.clienteling_actions` holds one row per client at 30 June 2019: the champion action and reason, the challenger action, predicted uplift, and a `predicted_harmful` flag.

| Reason (champion) | Clients |
|---|---:|
| Below this advisor's capacity cut-off | 193,573 |
| No personalisation consent | 174,768 |
| Among this advisor's most valuable clients | 104,869 |
| Not active in the last two years | 25,401 |

Consent is a hard gate and "no action" is an explicit, explained outcome. The list is decision support for advisors, not automatic outreach.

## Product picks

Each contacted client gets picks from the current collection at their home store: pieces launched in the 8 weeks before the cut-off or due in the 4 weeks after, excluding anything they already own. Availability comes from the store's planned assortment, which is known in advance; no future sales are used. Picks are judged on **real** purchases in the next 90 days, among clients who bought at least one candidate piece. Blend weights were tuned on the 2018 pilot (4,095 combinations) and tested once on 2019.

Signals, all known at the cut-off: chain and home-store popularity, the client's category, colour and fabric affinity, and **visual similarity**: the cosine similarity between a piece's image embedding and the client's last 10 purchases, using the Marqo-FashionSigLIP embeddings from the sibling **Silhouette Vision** project.

**Whole current collection** (32,289 clients):

| Method | Bought one of the top 5 | Top 10 |
|---|---:|---:|
| Random | 6.3% | 12.2% |
| Category / colour / fabric affinity alone | 5.7–7.7% | 11.9–14.7% |
| Chain popularity | 10.6% | 19.1% |
| **Visual similarity alone** | **10.5%** | 18.4% |
| Blend without visual | 12.4% | 20.3% |
| Blend with visual | 12.5% | 20.1% |

Visual similarity on its own matches popularity, but adds nothing measurable once popularity is in the blend: +0.1 points [−0.2, +0.4].

**New launches only** (no sales yet, so popularity cannot rank them; 9,224 clients who bought one):

| Method | Bought one of the top 5 | Top 10 |
|---|---:|---:|
| Random | 24.9% | 45.7% |
| Category / colour / fabric affinity | 31.0% | 51.5% |
| Visual similarity alone | 30.8% | 52.6% |
| **Affinity + visual** | **31.8%** | **53.1%** |

For new launches, visual similarity does add a small, statistically significant gain: **+0.9 points [0.04, 1.7]** over affinity alone. That is where it belongs in a luxury house: every new collection starts with no sales history, and the image is the richest thing known about a piece before launch.
