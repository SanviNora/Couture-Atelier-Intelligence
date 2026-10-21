# Dashboard

A seven-page Streamlit app that reads only `public_bundle/`: aggregates, model outputs, synthetic examples and public data. The hosted demo and a local run therefore show exactly the same thing, and no raw client row can reach it.

```bash
python scripts/build_public_bundle.py      # refresh the bundle from reports/ and the warehouse (guard-checked)
streamlit run dashboard/app.py             # http://localhost:8501
```

## Pages

| Page | For | What it shows |
|---|---|---|
| Decision brief | Executive | Four headline results and the elevation chart |
| Client intelligence | Client development | Model-minus-baseline with 95% intervals, value tiers, what drives each model, the 9-calibration sensitivity, the COVID stress test |
| Clienteling | Advisors and their managers | Champion vs challenger across pilot draws, advisor capacity, the experiment design, an illustrative advisor book, product picks on real purchases |
| Demand and allocation | Planners | Allocation rules as supply grows, the scarcity trap, launch-forecast error, demand recovery, hero pieces to clients |
| Brand Tier Dial | Executive | Profit by brand policy with lift scenario and brand-cost controls, the value of an exit route, market context |
| Model health | Data science | Input drift (PSI) between scoring dates and outcome monitoring |
| Method and gaps | Reviewers | The full gaps log with filters, and where each data source comes from |

## Design rules

- **Colour**: four categorical slots (jade, gold, sapphire, oxblood) validated for colour-vision deficiency on the page surface, adjacent and all pairs. The project's original palette failed (gold and coral indistinguishable under deuteranopia; three colours too grey), so it was re-stepped and re-validated.
- **Emphasis over rainbow**: one or two marks that carry the point are coloured; the rest are grey.
- **Every chart has a "Show data" table**, so no value depends on hover or colour alone; gold sits below 3:1 contrast, which is why labels and tables are always present.
- **No dual axes; values directly labelled only where they matter; status uses icon + label, never colour alone.**
- Checked by rendering every page in a real browser at desktop and phone width (Playwright), not only by tests.

## Public bundle

`scripts/build_public_bundle.py` copies the reports and adds:

| File | Provenance | Contents |
|---|---|---|
| `drift.csv` | aggregate | Population stability index of 8 model inputs between consecutive scoring dates |
| `advisor_reasons.csv`, `champion_challenger_overlap.csv` | aggregate | Clients by tier and action, groups of 10 or more only |
| `forecast_by_category.csv` | aggregate | Launch forecast vs true demand by category |
| `demo_advisor_book.csv` | synthetic | 40 illustrative clients, each attribute drawn independently from tier-level deciles |
| `gaps_log.csv` | documentation | `docs/gaps_and_decisions.md` as a table |

Every file is declared in `MANIFEST.json`. The bundle is built in a staging folder and only replaces `public_bundle/` if `check_bundle` passes (no real client identifiers, raw timestamps or aggregate groups under 10).

## Deploying on Streamlit Community Cloud

1. The GitHub repository must contain `dashboard/app.py`, `dashboard/requirements.txt`, `.streamlit/config.toml` and `public_bundle/`. Raw data is never committed (`.gitignore`), so the repository is safe to make public.
2. Sign in at share.streamlit.io with GitHub and choose **Create app** → repository `SanviNora/Couture-Atelier-Intelligence`, branch `main`, main file `dashboard/app.py`.
3. Under Advanced settings pick Python 3.12 or later; the app installs `dashboard/requirements.txt`.
4. After every change to the analysis, rebuild the bundle, commit it, and the app redeploys on push.
