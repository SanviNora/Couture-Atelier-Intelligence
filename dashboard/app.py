"""Couture Atelier Intelligence: decision dashboard.

Reads only ``public_bundle/`` (aggregates, model outputs, synthetic examples
and public data), so the hosted demo and a local run show the same thing.
Rebuild the bundle with ``python scripts/build_public_bundle.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BUNDLE = Path(__file__).resolve().parents[1] / "public_bundle"

# Categorical slots, validated for colour-vision deficiency on the page surface (all pairs, contrast >= 3:1).
LAVENDER, HONEY, CHERRY, TEAL = "#5b4fe0", "#b88200", "#d2384f", "#009180"
CONTEXT = "#d9d3c8"  # de-emphasised marks
INK, INK_2, MUTED, GRID, AXIS = "#16151a", "#4f4c55", "#75717a", "#eee9e0", "#d6d0c4"
STATUS = {"stable": ("#0ca30c", "✓"), "watch": ("#fab219", "●"), "act": ("#d03b3b", "▲")}
FONT = 'Geist, system-ui, -apple-system, "Segoe UI", sans-serif'
REPO = "https://github.com/SanviNora/Couture-Atelier-Intelligence"
FEATURE_LABELS = {
    "spend_12m": "Spend, last 12 months", "visits_12m": "Visits, last 12 months", "recency_weeks": "Weeks since last visit",
    "lifetime_visits": "Visits, all time", "products_12m": "Different pieces bought, 12 months",
    "full_price_share_12m": "Share bought at full price", "p_alive_mbg": "Probability still active (MBG/NBD)",
    "expected_visits_mbg": "Expected visits next year (MBG/NBD)", "expected_visits_bg": "Expected visits (BG/NBD)",
    "p_alive_bg": "Probability still active (BG/NBD)", "max_visit_value_12m": "Largest visit, 12 months",
    "mean_visit_value_12m": "Average visit, 12 months", "spend_180d": "Spend, last 6 months",
    "spend_90d": "Spend, last 3 months", "visits_90d": "Visits, last 3 months", "spend_momentum": "Share of spend in last 6 months",
    "pred_spend_mbg": "Probabilistic spend forecast", "spend_per_visit_gg": "Expected spend per visit",
    "categories_12m": "Categories bought, 12 months", "stores_12m": "Stores visited, 12 months",
    "hero_share_12m": "Share of hero pieces", "last_visit_value": "Last visit value", "ticket_trend": "Ticket-size trend",
    "holiday_share": "Share of visits in Nov–Dec", "mean_gap_weeks": "Average weeks between visits",
    "cutoff_month": "Month of scoring date", "tenure_weeks_capped": "Weeks as a client (capped at 52)",
}


def readable(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def padded(values, share: float = 0.22) -> list[float]:
    """Axis range with headroom so value labels at the bar ends are never clipped."""
    low, high = min(0.0, min(values)), max(0.0, max(values))
    span = (high - low) or 1.0
    return [low - share * span if low < 0 else 0, high + share * span]



# ---- data -------------------------------------------------------------------------------
@st.cache_data
def report(name: str) -> dict:
    return json.loads((BUNDLE / name).read_text())


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(BUNDLE / name)


# ---- presentation helpers -------------------------------------------------------------
def style() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 4rem; }
        h1 { font-size: clamp(2.5rem, 5vw, 3.6rem) !important; line-height: 1.02 !important;
             letter-spacing: -0.015em !important; padding-bottom: 0.4rem !important; }
        h1 em, h2 em { color: #5b4fe0; }
        h2, h3 { letter-spacing: -0.01em !important; }
        .eyebrow { display: inline-block; background: #ece9fd; color: #4436c9; border-radius: 999px;
                   padding: 0.28rem 0.8rem; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
                   text-transform: uppercase; margin-bottom: 0.6rem; }
        .lede { color: #4f4c55; font-size: 1.06rem; line-height: 1.55; max-width: 68ch; margin-bottom: 1.6rem; }
        .tile { border: 1px solid #e8e2d7; border-radius: 20px; padding: 1.15rem 1.25rem 1.1rem; background: #ffffff;
                height: 100%; }
        .tile .value { font-family: 'Instrument Serif', Georgia, serif; font-size: 2.7rem; color: #16151a;
                       line-height: 1; letter-spacing: -0.01em; }
        .tile .label { color: #16151a; font-size: 0.9rem; font-weight: 500; margin-top: 0.55rem; }
        .tile .note { color: #75717a; font-size: 0.78rem; margin-top: 0.3rem; }
        .chip { display: inline-block; padding: 0.22rem 0.7rem; margin: 0 0.3rem 0.4rem 0; border-radius: 999px;
                font-size: 0.8rem; font-weight: 500; color: #16151a; border: 1.5px solid; background: #ffffff; }
        .prov { color: #75717a; font-size: 0.78rem; line-height: 1.5; margin-top: 1rem; }
        .footer { color: #75717a; font-size: 0.78rem; border-top: 1px solid #e8e2d7; margin-top: 3rem;
                  padding-top: 1rem; }
        .footer a { color: #4436c9; text-decoration: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lede">{lede}</div>', unsafe_allow_html=True)


def tiles(items: list[tuple[str, str, str]]) -> None:
    for column, (value, label, note) in zip(st.columns(len(items)), items, strict=True):
        column.markdown(f'<div class="tile"><div class="value">{value}</div><div class="label">{label}</div>'
                        f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def provenance(text: str) -> None:
    st.markdown(f'<div class="prov">{text}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="footer">Couture Atelier Intelligence · built by Sanvi · '
                f'<a href="{REPO}">source on GitHub</a> · aggregates, model outputs and synthetic examples only</div>',
                unsafe_allow_html=True)


def chart(fig: go.Figure, data: pd.DataFrame, height: int = 340, key: str | None = None) -> None:
    has_legend = sum(1 for trace in fig.data if trace.showlegend is not False) > 1
    fig.update_layout(
        height=height + (36 if has_legend else 0),
        margin={"l": 8, "r": 8, "t": 78 if has_legend else 44, "b": 8}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font={"family": FONT, "color": INK_2, "size": 12},
        legend={"orientation": "h", "x": 0, "y": 1.0, "yanchor": "bottom", "font": {"color": INK_2}},
        barcornerradius=4, bargap=0.35, hoverlabel={"font_family": FONT},
        title={"font": {"family": FONT, "size": 14, "color": INK, "weight": 600}, "x": 0.005, "xref": "container", "xanchor": "left",
               "y": 0.985, "yref": "container", "yanchor": "top"},
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont={"color": MUTED}, title_font={"color": INK_2})
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS, tickfont={"color": MUTED},
                     title_font={"color": INK_2})
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, key=key)
        with st.expander("Show data"):
            st.dataframe(data, hide_index=True, use_container_width=True)


def emphasis_bar(labels: list[str], values: list[float], highlight: set[str], fmt: str, title: str,
                 horizontal: bool = True, reference: tuple[float, str] | None = None) -> go.Figure:
    colours = [LAVENDER if label in highlight else CONTEXT for label in labels]
    show = fmt if callable(fmt) else fmt.format
    text = [show(v) if label in highlight else "" for label, v in zip(labels, values, strict=True)]
    hover = [f"{label}: {show(v)}" for label, v in zip(labels, values, strict=True)]
    kwargs = {"y": labels, "x": values, "orientation": "h"} if horizontal else {"x": labels, "y": values}
    fig = go.Figure(go.Bar(marker_color=colours, text=text, textposition="outside", hovertext=hover,
                           hoverinfo="text", cliponaxis=False, textfont={"color": INK}, **kwargs))
    if reference:
        value, name = reference
        line = {"x0": value, "x1": value, "y0": 0, "y1": 1, "yref": "paper"} if horizontal else \
            {"y0": value, "y1": value, "x0": 0, "x1": 1, "xref": "paper"}
        fig.add_shape(type="line", line={"color": INK_2, "width": 1}, **line)
        fig.add_annotation(text=name, showarrow=False, font={"color": INK_2, "size": 11},
                           **({"x": value, "y": 1.04, "yref": "paper", "xanchor": "left", "xshift": 4}
                              if horizontal else
                              {"x": 1, "xref": "paper", "y": value, "yshift": 10}))
    fig.update_layout(title=title, showlegend=False)
    value_range = padded(values)
    if horizontal:
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=GRID, range=value_range)
    else:
        fig.update_yaxes(range=value_range)
    return fig


def money(value: float) -> str:
    return f"€{value:,.1f}M"


def signed_money(value: float) -> str:
    """+€42M, −€48M, €0M: sign before the currency, with a real minus sign."""
    if round(value) == 0:
        return "€0M"
    return f"{'+' if value > 0 else '−'}€{abs(value):,.0f}M"


def minus(text: str) -> str:
    return text.replace("-", "−")


# ---- pages ------------------------------------------------------------------------------
def executive() -> None:
    header("Couture Atelier · decision brief", "Sharper decisions, a *quieter* brand",
           "A luxury retail decision system built on 665,000 real fashion clients and 110 stores (Visuelle 2.0), "
           "with luxury prices, client tiers and scarcity calibrated to published industry figures. "
           "Every number below is read from the project's reports.")
    dial = report("tier_dial.json")["scenarios"]["central"]
    elevation = pd.DataFrame(dial["elevation"])
    keep_outlet = elevation[elevation["outlet_exit"] & (elevation["weeks_with_markdowns_kept"] == 0)].iloc[0]
    today = elevation[elevation["weeks_with_markdowns_kept"] == 12].iloc[0]
    luxury = elevation[~elevation["outlet_exit"]].iloc[0]
    alloc = pd.DataFrame(report("allocation.json")["store_allocation"]).set_index("policy")["revenue_eur_m"]
    ps = report("price_strategy.json")
    ten = next(row for row in ps["cut"]["cuts"] if round(row["cut"], 2) == 0.10)
    gap = ps["cohort"]["entry_minus_higher_within_boutique_month"]["returned_2y"]
    tiles([
        (f"+{ten['breakeven_volume_gain']:.0%} vs +{ten['volume_gain_sustained']:.0%}",
         "volume a 10% price cut needs, vs what real demand gives", "so ladder down, don't cut"),
        (minus(f"{(keep_outlet['gross_profit_eur_m'] / today['gross_profit_eur_m'] - 1) * 100:+.1f}%"),
         "profit change from dropping every markdown", "markdown lift fades within weeks"),
        (money(keep_outlet["gross_profit_eur_m"] - luxury["gross_profit_eur_m"]),
         "what refusing any exit for unsold stock costs", "closing the outlet / pre-owned route"),
        (f"+{money(alloc['optimal_with_uncertainty'] - alloc['retailer_opening_share'])}",
         "forecast-based allocation vs the retailer's own split", "scarce launches at 60% of demand"),
    ])
    gap_text = (f"{abs(gap['difference']) * 100:.1f} points {'more' if gap['difference'] > 0 else 'less'} often than"
                if gap["ci_low"] > 0 or gap["ci_high"] < 0 else "just as often as")
    st.markdown("&nbsp;")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("What the evidence says")
        st.markdown(
            f"- **Ladder down; don't cut.** A permanent cut needs far more volume than real demand gives. Clients "
            f"whose first piece was entry-priced came back {gap_text} those who started higher.\n"
            "- **Stop discounting; keep an exit.** Dropping markdowns barely moves profit and lifts margin; the "
            "real cost of luxury discipline is having nowhere to send unsold stock. The EU destruction ban "
            "(July 2026) makes a non-destructive exit unavoidable anyway.\n"
            "- **Allocate scarce launches by forecast, and hero pieces by client.** Using a forecast beats the "
            "retailer's rule; for hero pieces the store split hardly matters, but which clients get them does.\n"
            "- **Keep calling your best clients.** Uplift targeting did not reliably beat it across pilot draws, "
            "so it runs as a randomised challenger, measured on visit rate.\n"
            "- **Recover demand lost to stock-outs** before forecasting scarce pieces, or every season's buy "
            "shrinks a little more.")
    with right:
        steps = elevation.copy()
        steps["step"] = ["Today: 12 wks", "6 wks", "4 wks", "2 wks", "No markdowns", "No outlet"]
        fig = emphasis_bar(steps["step"].tolist(), steps["gross_profit_eur_m"].tolist(),
                           {"Today: 12 wks", "No markdowns", "No outlet"}, "€{:,.0f}M",
                           "Gross profit as a brand cuts markdowns, then closes its outlet")
        chart(fig, steps[["step", "gross_profit_eur_m", "profit_margin", "full_price_share_of_units_sold"]])
    provenance("Demand and client behaviour: real (Visuelle 2.0, 2016–2021). Prices, tiers, consent and contact "
               "effects: calibrated or simulated, every assumption in configs/. Search interest: Google Trends.")


def clients() -> None:
    header("Client intelligence", "Who matters *next* year",
           "Value, return and VIC-upgrade models chosen on pre-test data and tested once on 2019, with "
           "paired-bootstrap intervals. 2020 is a COVID stress test.")
    cv = report("client_value.json")
    test = cv["evaluation"]["test"]
    ci = pd.DataFrame([{"comparison": k.replace("_", " "), **v} for k, v in test["paired_bootstrap_95ci"].items()])
    labels = {"spend spearman ml minus naive": "Spend ranking (Spearman) vs last year's spend",
              "top1pct precision ml minus naive": "Next year's VICs found in top 1% vs last year's spend",
              "return auc ml minus recency": "Return next year (AUC) vs recency alone",
              "upgrade avg precision ml minus naive": "VIC upgrade (avg. precision) vs current spend"}
    ci["comparison"] = ci["comparison"].map(labels)
    fig = go.Figure(go.Scatter(
        x=ci["difference"], y=ci["comparison"], mode="markers", marker={"size": 10, "color": LAVENDER},
        error_x={"type": "data", "symmetric": False, "array": ci["ci_high"] - ci["difference"],
                 "arrayminus": ci["difference"] - ci["ci_low"], "color": LAVENDER, "thickness": 2, "width": 0},
        hovertemplate="%{y}<br>model minus baseline: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_2, line_width=1)
    fig.update_layout(title="Model minus baseline on the 2019 test, with 95% intervals (right of 0 = better)",
                      showlegend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    chart(fig, ci, height=300)

    sens = table("sensitivity_client.csv")
    sens["variant"] = sens.apply(lambda r: f"VIC share {r.vic_share_target:.0%}, cap {r.trade_up_cap}", axis=1)
    sens["spearman_gain"] = (sens["spend_spearman_ml"] - sens["spend_spearman_naive"]).round(3)
    diag = test["spend_bias_diagnosis"]
    tiles([
        (f"{int((sens['spearman_gain'] > 0).sum())} of {len(sens)}",
         "luxury-layer calibrations where the model still beats last year's spend",
         (f"VIC share {sens['vic_share_target'].min():.0%}–{sens['vic_share_target'].max():.0%}, "
          f"trade-up cap {sens['trade_up_cap'].min()}–{sens['trade_up_cap'].max()} visits")),
        (f"{sens['return_auc_ml'].min():.3f}–{sens['return_auc_ml'].max():.3f}", "return AUC across those calibrations",
         "visits are real data, so assumptions barely move it"),
        ((f"{test['spend']['selected_ml']['total_bias_pct']:+.0f}% → "
          f"{diag['selected_ml_trend_level']['total_bias_pct']:+.0f}%"),
         "2019 spend total, before and after the return-rate trend", "found after the test: a diagnosis, not a result"),
    ])
    st.markdown("&nbsp;")
    rates = list(diag["training_origin_return_rates"].values())
    st.markdown(
        f"**Why the spend total ran high.** Fewer clients came back each year: {rates[0]:.0%} → {rates[-1]:.0%} "
        f"across the 2017 training dates, then {diag['test_return_rate']:.1%} in 2019. Projecting that trend with "
        "data known at the end of 2018 removes the bias; client rankings, which drive every decision here, "
        "were unaffected.")
    importance = table("client_feature_importance.csv")
    top = importance[importance["model"] == "return (logistic)"].nlargest(8, "importance").copy()
    top["feature"] = top["feature"].map(readable)
    fig = go.Figure(go.Bar(y=top["feature"], x=top["importance"], orientation="h", marker_color=TEAL,
                           error_x={"type": "data", "array": top["std"], "color": INK_2, "thickness": 1}))
    fig.update_layout(title="What drives the return model: AUC lost when a feature is shuffled")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    chart(fig, top[["feature", "importance", "std"]], key="importance_chart")
    with st.expander("Spend-ranking gain in each of the 9 calibrations"):
        st.dataframe(sens[["variant", "spearman_gain", "spend_spearman_ml", "spend_spearman_naive", "return_auc_ml",
                           "upgrade_ap_ml", "upgrade_ap_naive"]], hide_index=True, use_container_width=True)
    provenance("Client behaviour real; euro values from the calibrated luxury layer (VICs hold 30% of revenue). "
               "The COVID stress test is on the model-health page.")


def clienteling() -> None:
    header("Clienteling", "Whom each advisor *calls* this quarter",
           "Calling each advisor's most valuable clients is the champion. A trusted uplift model runs as a "
           "randomised challenger. Contact effects are simulated from a documented rule because no public data "
           "records luxury outreach.")
    cl = report("clienteling.json")
    robust = cl["robustness_across_pilot_draws"]
    draws = pd.DataFrame(robust["draws"])
    power = cl["power_analysis_for_challenger_contacts"]["visit_rate"]
    tiles([
        (f"{robust['status_quo_share']:.0%} vs {robust['mean_with_trust_check']:.0%}",
         "champion vs uplift challenger: share of the best achievable euros",
         f"uplift trusted in {int(draws['trusted'].sum())} of {len(draws)} pilot draws"),
        (f"+{power['true_effect_per_contact'] * 100:.1f} pts", "effect of a contact on the chance of a visit",
         "euros are too noisy to measure"),
        (f"{power['holdout_share_client_randomised_cuped']:.0%}", "client holdout needed", "with CUPED, 80% power"),
        (f"{power['holdout_share_advisor_randomised_cuped']:.0%}", "of advisors, if randomising by advisor",
         f"design effect {power['design_effect_advisor_randomised']}"),
    ])
    st.markdown("&nbsp;")
    sweep = pd.DataFrame(cl["capacity_sweep"])
    fig = go.Figure()
    for policy, colour, name in (("status_quo_best_clients", LAVENDER, "Champion: best clients"),
                                 (cl["challenger_policy"], HONEY, "Uplift challenger"),
                                 ("random", CONTEXT, "Random")):
        part = sweep[sweep["policy"] == policy]
        fig.add_trace(go.Scatter(x=part["contacts_per_advisor"], y=part["share_of_oracle"], mode="lines+markers",
                                 name=name, line={"color": colour, "width": 2}, marker={"size": 8}))
    fig.update_layout(title="Share of the best achievable incremental euros as advisor capacity grows")
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title="contacts per advisor per quarter")
    chart(fig, sweep)
    st.markdown("&nbsp;")
    st.subheader("An advisor's book (illustrative synthetic clients)")
    book = table("demo_advisor_book.csv")
    st.dataframe(book, hide_index=True, use_container_width=True, column_config={
        "client": "Client", "tier": "Tier", "favourite_category": "Favourite category",
        "weeks_since_last_visit": st.column_config.NumberColumn("Weeks since last visit"),
        "spend_last_12m_eur": st.column_config.NumberColumn("Spend, 12 months", format="euro"),
        "predicted_contact_value_eur": st.column_config.NumberColumn("Predicted contact value", format="euro"),
        "champion_action": "Champion: best clients", "challenger_action": "Challenger: uplift",
        "advisor_note": st.column_config.TextColumn("Reason to reach out", width="large"),
        "note_source": "Note source",
    }, column_order=["client", "tier", "advisor_note", "champion_action", "challenger_action", "spend_last_12m_eur",
                     "weeks_since_last_visit", "favourite_category", "predicted_contact_value_eur", "note_source"])
    provenance("Synthetic clients drawn from tier-level deciles; no real client appears. Consent is a hard gate "
               "and 'no action' is an explained outcome. Notes may be drafted by a language model offline, but one "
               "is shown only if every fact in it matches the client's row and it avoids discount language; "
               "otherwise the advisor sees the template.")

    st.subheader("Product picks, judged on real purchases")
    rec = report("recommendations.json")
    allp, new = rec["test"], rec["test_new_launches_only"]
    nl = {"random": "Random", "affinity_only": "Category/colour/fabric affinity", "visual_only":
          "Visual similarity alone", "affinity_plus_visual": "Affinity + visual"}
    values = [new[k]["hit@5"] for k in nl]
    whole = pd.DataFrame({"pieces": "whole collection", "method": ["Random", "Blend with visual"],
                          "hit@5": [allp["random"]["hit@5"], allp["blend_with_visual"]["hit@5"]]})
    chart(emphasis_bar(list(nl.values()), values, {"Affinity + visual"}, "{:.1%}",
                       "New launches (no sales yet): clients who bought one of their top 5 picks"),
          pd.concat([pd.DataFrame({"pieces": "new launches", "method": list(nl.values()), "hit@5": values}), whole]),
          height=280, key="recs_new")
    gain = new["visual_gain_hit@5"]
    provenance(f"Image embeddings from the Silhouette Vision project. On new launches visual similarity adds "
               f"{gain['difference'] * 100:+.1f} points [{gain['ci_low'] * 100:.1f}, {gain['ci_high'] * 100:.1f}]; "
               f"across the whole collection the best blend reaches {allp['blend_with_visual']['hit@5']:.1%} "
               f"vs {allp['random']['hit@5']:.1%} at random.")


def planning() -> None:
    header("Demand and allocation", "Where *scarce* pieces should go",
           "Demand recovered from stock-outs, a store-level launch forecast with uncertainty, and allocation "
           "judged on true demand for 890 launches (Jul–Nov 2019).")
    demand = report("demand.json")
    alloc = report("allocation.json")
    scarce = report("scarcity_scenario.json")
    left, right = st.columns(2)
    with left:
        sweep = pd.DataFrame(alloc["supply_sweep"])
        fig = go.Figure()
        for policy, colour, name in (("optimal_with_uncertainty", LAVENDER, "Forecast-based (optimal)"),
                                     ("equal_split", HONEY, "Equal split"),
                                     ("retailer_opening_share", CHERRY, "Retailer's opening split"),
                                     ("oracle_true_demand", TEAL, "Oracle (true demand)")):
            part = sweep[sweep["policy"] == policy]
            fig.add_trace(go.Scatter(x=part["supply_share_of_forecast"], y=part["revenue_eur_m"],
                                     mode="lines+markers", name=name, line={"color": colour, "width": 2},
                                     marker={"size": 8}))
        fig.update_layout(title="Revenue by allocation rule as supply grows (€M)")
        fig.update_xaxes(tickformat=".0%", title="units available, share of forecast demand")
        chart(fig, sweep)
    with right:
        planners = pd.DataFrame(scarce["planners"]).T.reset_index(names="planner")
        planners["planner"] = planners["planner"].map({"treats_sales_as_demand": "Treats capped sales as demand",
                                                       "recovers_demand_first": "Recovers demand (one pass)",
                                                       "recovers_demand_with_em": "Recovers demand with EM"})
        fig = emphasis_bar(planners["planner"].tolist(), planners["demand_lost_to_stockouts_pct"].tolist(),
                           {"Recovers demand with EM", "Treats capped sales as demand"}, "{:.1f}%",
                           "Luxury scarcity trap: demand lost next season")
        chart(fig, planners)
    forecast = demand["launch_forecast"]
    chosen = forecast["models"][forecast["chosen_on_validation"]]["test"]
    baseline = forecast["models"]["store_category_average"]["test"]
    recov = demand["recovery"]["validation_artificial_censoring"]["censor_30pct_of_weeks"]
    tiles([
        (f"{chosen['wape']:.3f}", "launch forecast error (WAPE), Jul–Nov 2019",
         f"vs {baseline['wape']:.3f} for a store × category average"),
        (f"{forecast['quantile_interval_test']['coverage_q10_q90']:.1%}", "of outcomes inside the 80% interval",
         "the uncertainty is honest"),
        (minus(f"{recov['conditional']['total_bias_pct']:+.1f}%"), "bias recovering demand in sold-out weeks",
         minus(f"vs {recov['naive']['total_bias_pct']:+.0f}% if sales are taken as demand")),
    ])
    st.markdown("&nbsp;")
    st.subheader("Hero pieces: which clients get them")
    ca = report("client_allocation.json")
    policies = pd.DataFrame(ca["policies"]).T.reset_index(names="policy")
    labels = {"revenue_only": "Revenue only", "vic_first": "VICs first", "relationship_value": "Relationship value",
              "relationship_value_with_pipeline_reserve": "Relationship value + 30% for rising clients (chosen)"}
    policies["rule"] = policies["policy"].map(labels)
    fig = go.Figure()
    for column, colour, name in (("expected_revenue_eur_m", LAVENDER, "Revenue this season"),
                                 ("expected_relationship_value_eur_m", HONEY, "Relationship value")):
        fig.add_trace(go.Bar(y=policies["rule"], x=policies[column], orientation="h", name=name, marker_color=colour,
                             hovertemplate="%{y}<br>" + name + ": €%{x:.1f}M<extra></extra>"))
    totals = policies["expected_total_value_eur_m"]
    fig.add_trace(go.Scatter(y=policies["rule"], x=totals, mode="text", text=[f"  €{v:.0f}M" for v in totals],
                             textposition="middle right", showlegend=False, hoverinfo="skip",
                             textfont={"color": INK}))
    fig.update_layout(barmode="stack", title="Expected value of 15 hero launches by offer rule (€M)",
                      legend={"traceorder": "normal"})
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=padded(totals.tolist(), 0.15))
    chart(fig, policies[["rule", "expected_revenue_eur_m", "expected_relationship_value_eur_m",
                         "expected_total_value_eur_m", "share_offers_to_vic", "share_offers_to_rising"]],
          height=300, key="hero")
    with st.expander("How the choice moves with the relationship-value assumption"):
        st.dataframe(pd.DataFrame(ca["relationship_value_sensitivity"]), hide_index=True, use_container_width=True)
    provenance("Integer programme per boutique: one hero piece per client, consent required, reseller-risk "
               "clients excluded, 30% reserved for rising clients. Acceptance and relationship value are "
               "assumptions.")


def tier_dial() -> None:
    header("Pricing and brand tiers", "Cut, ladder or *hold*?",
           "In 2026 houses answered price fatigue three ways: Gucci cut list prices, Chanel and Dior added entry "
           "pieces under their icons, Hermès held. Client behaviour and the price response are measured on real "
           "data; margins and the size of an entry line are assumptions.")
    ps = report("price_strategy.json")
    cuts = pd.DataFrame(ps["cut"]["cuts"])
    ten = cuts[cuts["cut"].round(2) == 0.10].iloc[0]
    gap = ps["cohort"]["entry_minus_higher_within_boutique_month"]["returned_2y"]
    ladder = pd.DataFrame(ps["ladder"]["scenarios"])
    mid = ladder[ladder["cannibalisation"].round(2) == 0.25].iloc[0]
    tiles([
        (f"+{ten['breakeven_volume_gain']:.0%} vs +{ten['volume_gain_sustained']:.0%}",
         "volume a 10% list-price cut needs to break even, vs what it gets",
         minus(f"sustained response to real markdowns (elasticity {ps['cut']['elasticity']['sustained']:.1f})")),
        (minus(f"{gap['difference'] * 100:+.1f} pts"), "return rate of clients whose first piece was entry-priced",
         minus(f"vs higher first pieces, same boutique and month [{gap['ci_low'] * 100:.1f}, {gap['ci_high'] * 100:.1f}]")),
        (money(mid["total_eur_m"]), "two-year profit from an entry line",
         f"{ps['ladder']['entry_units']:,} entry units, a quarter cannibalising core pieces"),
    ])
    st.markdown("&nbsp;")
    left, right = st.columns([1.1, 1])
    with left:
        rows = [(f"Cut {r.cut:.0%}", r.profit_change_sustained_eur_m) for r in cuts.itertuples()]
        rows += [(f"Ladder, {r.cannibalisation:.0%} cannibalised", r.total_eur_m) for r in ladder.itertuples()]
        rows += [("Hold", ps["hold"]["profit_change_eur_m"])]
        strategy = pd.DataFrame(rows, columns=["strategy", "profit_change_eur_m"])
        best = strategy.loc[strategy["profit_change_eur_m"].idxmax(), "strategy"]
        fig = emphasis_bar(strategy["strategy"].tolist(), strategy["profit_change_eur_m"].tolist(),
                           {best, "Cut 10%", "Hold"}, signed_money, "Profit change vs holding, luxury policy (€M)",
                           reference=(0, "hold"))
        chart(fig, strategy, height=330, key="strategy")
    with right:
        tiers = pd.DataFrame(ps["cohort"]["by_first_tier"])
        fig = emphasis_bar(tiers["first_tier"].str.capitalize().tolist(), tiers["returned_2y"].tolist(),
                           {"Entry"}, "{:.1%}", "Came back within two years, by price of the first piece",
                           horizontal=False)
        fig.update_yaxes(tickformat=".0%")
        chart(fig, tiers, height=330, key="cohort")
    provenance(f"{ps['cohort']['clients']:,} real clients whose first visit was in 2017, followed for two years "
               f"(before COVID). The first-two-weeks markdown response ({ps['cut']['lift_first_two_weeks']:.2f}×) is an "
               "upper bound, not a plan: it fades within weeks. Extra units come from stock already bought, the "
               "assumption most favourable to cutting.")

    st.subheader("Markdowns: what it costs to stop discounting")
    dial = report("tier_dial.json")
    left, right = st.columns(2)
    scenario = left.segmented_control("Markdown lift", ["central", "conservative"], default="central",
                                      key="lift_scenario") or "central"
    cost = right.segmented_control("Brand cost per discounted unit", ["0%", "5%", "10%", "20%"], default="0%",
                                   key="brand_cost") or "0%"
    res = dial["scenarios"][scenario]
    policies = pd.DataFrame(res["policies"])
    column = f"profit_after_brand_cost_{cost.rstrip('%')}pct_eur_m"
    names = {"luxury": "Luxury", "aspirational": "Aspirational", "premium_optimised_markdowns": "Premium",
             "retailer_actual_markdowns": "Retailer's actual"}
    colours = {"Luxury": LAVENDER, "Aspirational": HONEY, "Premium": TEAL, "Retailer's actual": CHERRY}
    policies["name"] = policies["policy"].map(names)
    fig = go.Figure(go.Bar(x=policies["name"], y=policies[column], marker_color=[colours[n] for n in policies["name"]],
                           text=[f"€{v:,.0f}M" for v in policies[column]], textposition="outside",
                           cliponaxis=False, textfont={"color": INK}))
    fig.update_layout(title=f"Gross profit after a {cost} brand cost per discounted unit ({scenario} lift)",
                      showlegend=False, bargap=0.45)
    fig.update_yaxes(range=padded(policies[column].tolist(), 0.12), title="€M")
    chart(fig, policies[["name", "units_bought", "full_price_share_of_units_sold", "sell_through", "revenue_eur_m",
                         "gross_profit_eur_m", column, "profit_margin"]], height=300)
    exits = pd.DataFrame(res["exit_value_sensitivity"]).sort_values("exit_value_share_of_price")
    low, high = exits.iloc[0], exits.iloc[-1]
    tiles([
        (money(high["gross_profit_eur_m"] - low["gross_profit_eur_m"]),
         "profit an exit route is worth with no markdowns",
         (f"recovering {high['exit_value_share_of_price']:.0%} vs {low['exit_value_share_of_price']:.0%} of list "
          "price per unsold piece")),
        (f"{policies.loc[policies['policy'] == 'luxury', 'sell_through'].iloc[0]:.0%}",
         "sell-through at full price with a scarce buy", "the rest needs a non-destructive exit (EU ban, July 2026)"),
    ])
    st.markdown("&nbsp;")
    with st.expander("Markdown lift measured on real data (staggered difference-in-differences)"):
        effects = report("discount_effects.json")
        rows = pd.DataFrame(effects["markdown_lift"]["by_depth"])
        st.dataframe(rows[["depth", "events", "lift", "ci_low", "ci_high", "lift_weeks_3_4", "placebo_lift"]],
                     hide_index=True, use_container_width=True)
        st.dataframe(exits, hide_index=True, use_container_width=True)
    provenance("Shallow markdowns pass the placebo check; deeper ones do not (the retailer discounts where sales "
               "are already slipping), hence the conservative scenario. Discount-acquired clients were not worse "
               "in this data, so the brand cost of discounting is an explicit assumption.")


def health() -> None:
    header("Model health", "Would we have seen it *coming*?",
           "Input drift (population stability index against the previous scoring date) and outcome monitoring. "
           "PSI below 0.10 is stable, 0.10–0.25 worth watching, above 0.25 a reason to act.")
    drift = table("drift.csv")
    for comparison, part in drift.groupby("comparison", sort=False):
        st.markdown(f"**{comparison}**")
        cells = []
        for row in part.itertuples():
            colour, icon = STATUS[row.status]
            cells.append(f'<span class="chip" style="border-color:{colour}"><span style="color:{colour}">{icon}</span> '
                         f'{readable(row.feature)}: {row.psi:.2f} ({row.status})</span>')
        st.markdown(" ".join(cells), unsafe_allow_html=True)
    with st.expander("Show data"):
        st.dataframe(drift, hide_index=True, use_container_width=True)
    cv = report("client_value.json")["evaluation"]
    groups = drift.groupby("comparison", sort=False)["status"]
    acts, sizes = groups.apply(lambda x: int((x == "act").sum())).tolist(), groups.size().tolist()
    tiles([
        (f"{acts[0]} of {sizes[0]}", "inputs to act on between training and test", "explains the 2019 spend bias"),
        (f"{acts[1]} of {sizes[1]}", "inputs to act on before COVID", "the shock hit outcomes, not inputs"),
        (f"{cv['test']['lapse']['return_rate']:.1%} → {cv['covid']['lapse']['return_rate']:.1%}",
         "realised return rate, 2019 → 2020", "only outcome monitoring catches this"),
    ])
    st.markdown("&nbsp;")
    st.markdown("**Monitoring plan.** Track input PSI monthly; compare predicted with realised visits and spend "
                "every week; recalibrate spend totals on the latest completed year; fall back to "
                "last-year ranking when outcome bias passes 20%.")


def method() -> None:
    header("Method and gaps", "What is real, what is modelled, what is still *open*",
           "Every data issue, shortcut and judgment call found while building the project, with its status and "
           "the interview question it invites.")
    gaps = table("gaps_log.csv")
    left, right = st.columns(2)
    status = left.multiselect("Status", sorted(gaps["status"].unique()), key="gap_status")
    area = right.multiselect("Area", sorted(gaps["area"].unique()), key="gap_area")
    keep = gaps["status"].isin(status or gaps["status"].unique()) & gaps["area"].isin(area or gaps["area"].unique())
    view = gaps[keep]
    st.caption(f"{len(view)} of {len(gaps)} entries")
    st.dataframe(view, hide_index=True, use_container_width=True, height=460)
    st.subheader("Data")
    st.markdown(
        "- **Visuelle 2.0** (Skenderi et al., CVPR Workshops 2022): real clients, stores, weekly sales, discounts "
        "and restocks from one Italian fashion retailer, 2016–2021. Raw rows never leave the machine.\n"
        "- **Calibrated luxury layer**: price ladder, client tiers (VICs hold 30% of revenue), consent, advisors; "
        "every number cited or labelled as an assumption.\n"
        "- **Google Trends**, Oct 2021–Oct 2026: market context only.\n"
        "- **Silhouette Vision** image embeddings for product picks and launch forecasting.\n\n"
        "This dashboard reads only a public bundle of aggregates, model outputs and synthetic examples; an "
        "automated check blocks real client identifiers, raw timestamps and groups smaller than 10.")

    st.subheader("Market context")
    pulse = table("market_pulse.csv").dropna(subset=["change_pct"])
    pulse = pulse[pulse["source_file"].isin(["aspirational_bags", "luxury_bags", "luxury_brands"])
                  & (pulse["term"] != "Louis Vuitton Speedy")].sort_values("change_pct")
    fig = emphasis_bar(pulse["term"].tolist(), pulse["change_pct"].tolist(), {"Coach Tabby", "Gucci", "Hermes Birkin"},
                       "{:+.0f}%", "US search interest, last 12 months vs Oct 2021–Oct 2022 (Google Trends)",
                       reference=(0, ""))
    fig.update_yaxes(autorange=True)
    chart(fig, pulse[["source_file", "term", "first_year_mean", "last_year_mean", "change_pct"]], key="pulse")

PAGES = [
    (executive, "Decision brief", ":material/insights:"),
    (clients, "Client intelligence", ":material/diamond:"),
    (clienteling, "Clienteling", ":material/support_agent:"),
    (planning, "Demand and allocation", ":material/inventory_2:"),
    (tier_dial, "Pricing and tiers", ":material/sell:"),
    (health, "Model health", ":material/monitor_heart:"),
    (method, "Method and gaps", ":material/description:"),
]


def main() -> None:
    st.set_page_config(page_title="Couture Atelier Intelligence", page_icon="◆", layout="wide")
    style()
    pages = [st.Page(fn, title=title, icon=icon, default=(i == 0)) for i, (fn, title, icon) in enumerate(PAGES)]
    st.navigation(pages, position="top").run()
    footer()


if __name__ == "__main__":
    main()
