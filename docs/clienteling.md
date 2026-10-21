# Clienteling

Which clients each advisor should contact this quarter, judged by the sales a contact adds rather than by how many contacts are made, and which pieces to show them.

```bash
python scripts/build_clienteling.py       # pilot, uplift models, policies, power analysis (about 30 seconds)
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

## Choosing an uplift model when the pilot cannot tell them apart

| Learner (simplest first) | Qini on pilot holdout [95% CI] | Correlation with the true effect at deployment |
|---|---|---|
| **T-learner, logistic, core features** | 16.8 [−19.7, 68.0] | **0.32** |
| T-learner, logistic, all features | 37.9 [6.3, 88.3] | 0.34 |
| T-learner, gradient boosting | 45.2 [2.7, 106.7] | 0.09 |
| T-learner, gradient boosting, large leaves | **53.3** [21.4, 94.8] | 0.02 |
| Transformed outcome | 53.3 [11.5, 102.7] | 0.02 |
| X-learner on spend | −14.5 [−57.9, 36.8] | 0.03 |

The average contact effect is small (1.7 points on the chance of a visit), so the Qini intervals all overlap, and the highest Qini belongs to models that barely recover the true effect. The rule used: **take the simplest learner whose Qini interval reaches the best learner's Qini**. That picks the logistic T-learner on core features, which is also the one that recovers the effect best. Features could explain the effect up to a correlation of 0.71 (the rest is unobservable responsiveness), so the chosen model captures about half of what is achievable; a larger pilot is the main lever.

## Results at deployment

Contacts are ranked by predicted uplift × the value of a typical visit for that client, and nobody is contacted when the model predicts no lift.

| Policy (104,869 contacts) | True incremental € | € per contact | Share of best possible |
|---|---:|---:|---:|
| Random | €27.2M | €260 | 37.3% |
| Highest predicted spend if contacted | €63.3M | €603 | 86.7% |
| Churn risk × value | €63.8M | €608 | 87.3% |
| **"Call your best clients" (status quo)** | €63.8M | €609 | 87.4% |
| **Uplift policy (chosen)** | **€64.5M** | **€615** | **88.3%** |
| Oracle (true effect known) | €73.0M | €698 | 100% |

How the gap changes with advisor capacity:

| Contacts per advisor per quarter | Status quo | Uplift policy |
|---|---:|---:|
| 20 | **84.8%** | 83.6% |
| 40 | 87.4% | **88.3%** |
| 80 | 89.6% | **94.1%** |

What this means:

- **In euros, calling your best clients is a strong default.** A top client's typical visit is worth many times a mid-tier client's, so even a modest response from them adds a lot. This matches how luxury advisors already work.
- **Uplift helps at the margin, and more as capacity grows.** When advisors only have time for their top clients, there is little to gain. With more capacity, uplift finds persuadable mid-tier clients that a value ranking misses (+4.5 points at 80 contacts per advisor).
- **Harmful contacts are hard to avoid.** 13–17% of contacts under any practical policy reduce sales slightly, mostly because of the unobservable responsiveness. The model only catches the observable "do not disturb" pattern.

## Experiment design to confirm it in a real house

Among the clients the uplift policy would contact (from `reports/clienteling.json`):

| Success metric | Effect per contact | Spread | Holdout needed (client randomised, with CUPED) | Holdout needed (whole advisors held out, with CUPED) |
|---|---:|---:|---:|---:|
| Euros spent in 90 days | €615 | SD €63,500 | Not detectable even with a 50% holdout | Not detectable |
| **Visited in 90 days** | **+1.7 points** | SD 0.44 | **5%** | **22%** |

Recommendation: measure the pilot on **visit rate**, with euros as a secondary readout. Hold out about 5% of eligible clients at random, or about 22% of advisors if advisors would otherwise contact held-out clients anyway. Use last-12-month behaviour as a CUPED covariate. Randomising by advisor is safer operationally but needs a larger holdout because clients of the same advisor behave alike (design effect 3.5).

## Action list

`lux.clienteling_actions` holds one row per client at 30 June 2019 with the action and the reason:

| Reason | Clients |
|---|---:|
| Contact expected to add sales | 104,869 |
| Below this advisor's capacity cut-off | 193,507 |
| No personalisation consent | 174,768 |
| Not active in the last two years | 25,401 |
| Contact not expected to help; leave undisturbed | 66 |

Consent is a hard gate, and "no action" is an explicit, explained outcome. The list is decision support for advisors, not automatic outreach.

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
