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

# Categorical slots, validated for colour-vision deficiency on the page surface (adjacent and all pairs).
JADE, GOLD, SAPPHIRE, OXBLOOD = "#00866f", "#c99212", "#2f6bc0", "#b8402f"
CONTEXT = "#c9c7bf"  # de-emphasised marks
INK, INK_2, MUTED, GRID, AXIS = "#1a1c23", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STATUS = {"stable": ("#0ca30c", "✓"), "watch": ("#fab219", "●"), "act": ("#d03b3b", "▲")}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
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
        @import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:opsz,wght@6..96,500&display=swap');
        h1, h2 { font-family: 'Bodoni Moda', Didot, Georgia, serif !important; font-weight: 500 !important;
                 letter-spacing: 0 !important; }
        h1 { font-size: 2.3rem !important; }
        .eyebrow { color: #00866f; font: 700 0.72rem system-ui, sans-serif; letter-spacing: 0.14em;
                   text-transform: uppercase; margin-bottom: 0.2rem; }
        .lede { color: #52514e; font-size: 1.02rem; max-width: 70ch; margin-bottom: 1.2rem; }
        .tile { border: 1px solid #e1e0d9; border-radius: 6px; padding: 1rem 1.1rem; background: #ffffff;
                height: 100%; }
        .tile .value { font: 600 2rem system-ui, sans-serif; color: #1a1c23; line-height: 1.1; }
        .tile .label { color: #52514e; font-size: 0.86rem; margin-top: 0.35rem; }
        .tile .note { color: #898781; font-size: 0.76rem; margin-top: 0.35rem; }
        .chip { display: inline-block; padding: 0.1rem 0.55rem; margin: 0 0.25rem 0.35rem 0; border-radius: 999px;
                font-size: 0.8rem; font-weight: 500; color: #1a1c23; border: 1.5px solid; background: #ffffff; }
        .prov { color: #898781; font-size: 0.78rem; }
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


def chart(fig: go.Figure, data: pd.DataFrame, height: int = 340, key: str | None = None) -> None:
    has_legend = sum(1 for trace in fig.data if trace.showlegend is not False) > 1
    fig.update_layout(
        height=height + (36 if has_legend else 0),
        margin={"l": 8, "r": 8, "t": 78 if has_legend else 44, "b": 8}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font={"family": FONT, "color": INK_2, "size": 12},
        legend={"orientation": "h", "x": 0, "y": 1.0, "yanchor": "bottom", "font": {"color": INK_2}},
        barcornerradius=4, bargap=0.35, hoverlabel={"font_family": FONT},
        title={"font": {"size": 13, "color": INK}, "x": 0.005, "xref": "container", "xanchor": "left",
               "y": 0.985, "yref": "container", "yanchor": "top"},
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont={"color": MUTED}, title_font={"color": INK_2})
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS, tickfont={"color": MUTED},
                     title_font={"color": INK_2})
    st.plotly_chart(fig, use_container_width=True, key=key)
    with st.expander("Show data"):
        st.dataframe(data, hide_index=True, use_container_width=True)


def emphasis_bar(labels: list[str], values: list[float], highlight: set[str], fmt: str, title: str,
                 horizontal: bool = True, reference: tuple[float, str] | None = None) -> go.Figure:
    colours = [JADE if label in highlight else CONTEXT for label in labels]
    text = [fmt.format(v) if label in highlight else "" for label, v in zip(labels, values, strict=True)]
    hover = [f"{label}: {fmt.format(v)}" for label, v in zip(labels, values, strict=True)]
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


# ---- pages ------------------------------------------------------------------------------
def executive() -> None:
    header("Couture Atelier · decision brief", "Sharper decisions, a quieter brand",
           "A luxury retail decision system built on 665,000 real fashion clients and 110 stores (Visuelle 2.0), "
           "with luxury prices, client tiers and scarcity calibrated to published industry figures. "
           "Every number below is read from the project's reports.")
    dial = report("tier_dial.json")["scenarios"]["central"]
    elevation = pd.DataFrame(dial["elevation"])
    keep_outlet = elevation[elevation["outlet_exit"] & (elevation["weeks_with_markdowns_kept"] == 0)].iloc[0]
    today = elevation[elevation["weeks_with_markdowns_kept"] == 12].iloc[0]
    luxury = elevation[~elevation["outlet_exit"]].iloc[0]
    alloc = pd.DataFrame(report("allocation.json")["store_allocation"]).set_index("policy")["revenue_eur_m"]
    ci = report("client_value.json")["evaluation"]["test"]["paired_bootstrap_95ci"]["return_auc_ml_minus_recency"]
    auc = report("client_value.json")["evaluation"]["test"]["lapse"]["return_auc_ml"]
    tiles([
        (f"{(keep_outlet['gross_profit_eur_m'] / today['gross_profit_eur_m'] - 1) * 100:+.1f}%",
         "profit change from dropping every markdown", "markdown lift fades within weeks"),
        (money(keep_outlet["gross_profit_eur_m"] - luxury["gross_profit_eur_m"]),
         "what refusing any exit for unsold stock costs", "closing the outlet / pre-owned route"),
        (f"+{money(alloc['optimal_with_uncertainty'] - alloc['retailer_opening_share'])}",
         "forecast-based allocation vs the retailer's own split", "scarce launches at 60% of demand"),
        (f"{auc:.3f}", "return-next-year model (AUC)",
         f"+{ci['difference']:.3f} vs recency [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"),
    ])
    st.markdown("&nbsp;")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("What the evidence says")
        st.markdown(
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
    header("Client intelligence", "Who matters next year",
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
        x=ci["difference"], y=ci["comparison"], mode="markers", marker={"size": 10, "color": JADE},
        error_x={"type": "data", "symmetric": False, "array": ci["ci_high"] - ci["difference"],
                 "arrayminus": ci["difference"] - ci["ci_low"], "color": JADE, "thickness": 2, "width": 0},
        hovertemplate="%{y}<br>model minus baseline: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_2, line_width=1)
    fig.update_layout(title="Model minus baseline on the 2019 test, with 95% intervals (right of 0 = better)",
                      showlegend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    chart(fig, ci, height=300)

    left, right = st.columns(2)
    with left:
        tiers = pd.Series(cv["tier_counts"]).reindex(["top_tier", "vic", "core", "aspirational", "lapsed"])
        names = ["Top tier (0.1%)", "VIC (next 0.9%)", "Core (next 9%)", "Aspirational", "Lapsed"]
        fig = go.Figure(go.Bar(y=names, x=tiers.values, orientation="h", marker_color=SAPPHIRE,
                               text=[f"{v:,}" for v in tiers.values], textposition="outside", cliponaxis=False,
                               textfont={"color": INK}))
        fig.update_layout(title="Clients by value tier, 31 Dec 2019")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(range=padded(tiers.values.tolist()))
        chart(fig, pd.DataFrame({"tier": names, "clients": tiers.values}))
    with right:
        importance = table("client_feature_importance.csv")
        options = sorted(importance["model"].unique())
        model = st.segmented_control("Model", options, default=options[0], key="importance_model") or options[0]
        top = importance[importance["model"] == model].nlargest(8, "importance").copy()
        top["feature"] = top["feature"].map(readable)
        fig = go.Figure(go.Bar(y=top["feature"], x=top["importance"], orientation="h", marker_color=SAPPHIRE,
                               error_x={"type": "data", "array": top["std"], "color": INK_2, "thickness": 1}))
        fig.update_layout(title=f"What drives it: {top['metric'].iloc[0]} when a feature is shuffled")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        chart(fig, top[["feature", "importance", "std"]], key="importance_chart")

    st.subheader("Do the conclusions depend on the luxury-layer assumptions?")
    sens = table("sensitivity_client.csv")
    sens["variant"] = sens.apply(lambda r: f"VIC share {r.vic_share_target:.0%}, cap {r.trade_up_cap}", axis=1)
    sens["spearman_gain"] = sens["spend_spearman_ml"] - sens["spend_spearman_naive"]
    fig = emphasis_bar(sens["variant"].tolist(), sens["spearman_gain"].round(3).tolist(),
                       set(sens.loc[sens["is_baseline"], "variant"]), "{:+.3f}",
                       "Spend-ranking gain over the naive baseline, in each of 9 calibrations", reference=(0, "no gain"))
    chart(fig, sens[["variant", "spend_spearman_ml", "spend_spearman_naive", "return_auc_ml", "upgrade_ap_ml",
                     "upgrade_ap_naive"]], height=380)
    covid = cv["evaluation"]["covid"]["lapse"]
    tiles([
        (f"{test['lapse']['return_rate']:.1%} → {covid['return_rate']:.1%}", "share of clients returning, 2019 → 2020",
         "COVID store closures"),
        (f"{cv['evaluation']['covid']['spend']['naive_last_12m']['total_bias_pct']:+.0f}%",
         "naive spend forecast bias in 2020", "every model over-predicted"),
        (f"{covid['return_auc_ml']:.3f}", "return model AUC in 2020", "ranking held up"),
    ])
    provenance("Client behaviour real; euro values from the calibrated luxury layer (VICs hold 30% of revenue).")


def clienteling() -> None:
    header("Clienteling", "Whom each advisor calls this quarter",
           "Calling each advisor's most valuable clients is the champion. A trusted uplift model runs as a "
           "randomised challenger. Contact effects are simulated from a documented rule because no public data "
           "records luxury outreach.")
    cl = report("clienteling.json")
    draws = pd.DataFrame(cl["robustness_across_pilot_draws"]["draws"])
    status_quo = cl["robustness_across_pilot_draws"]["status_quo_share"]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[f"Draw {d}" for d in draws["draw"]], y=draws["uplift_share_of_oracle"],
                                 mode="markers+text", name="Uplift challenger", marker={"size": 11, "color": GOLD},
                                 text=[("trusted" if t else "flagged") for t in draws["trusted"]],
                                 textposition="bottom center", textfont={"color": INK_2}))
        fig.add_hline(y=status_quo, line_color=JADE, line_width=2,
                      annotation_text=f"Champion: best clients {status_quo:.1%}", annotation_font_color=INK)
        fig.update_layout(title="Share of the best achievable incremental euros, by pilot draw")
        fig.update_yaxes(tickformat=".0%", range=[0.5, 0.95])
        chart(fig, draws)
    with right:
        sweep = pd.DataFrame(cl["capacity_sweep"])
        fig = go.Figure()
        for policy, colour, name in (("status_quo_best_clients", JADE, "Champion: best clients"),
                                     (cl["challenger_policy"], GOLD, "Uplift challenger"),
                                     ("random", CONTEXT, "Random")):
            part = sweep[sweep["policy"] == policy]
            fig.add_trace(go.Scatter(x=part["contacts_per_advisor"], y=part["share_of_oracle"], mode="lines+markers",
                                     name=name, line={"color": colour, "width": 2}, marker={"size": 8}))
        fig.update_layout(title="Advisor capacity: contacts per quarter (first pilot draw)")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(title="contacts per advisor")
        chart(fig, sweep)

    power = cl["power_analysis_for_challenger_contacts"]["visit_rate"]
    tiles([
        (f"+{power['true_effect_per_contact'] * 100:.1f} pts", "effect of a contact on the chance of a visit",
         "euros are too noisy to measure"),
        (f"{power['holdout_share_client_randomised_cuped']:.0%}", "client holdout needed",
         "with CUPED, 80% power"),
        (f"{power['holdout_share_advisor_randomised_cuped']:.0%}", "of advisors, if randomising by advisor",
         f"design effect {power['design_effect_advisor_randomised']}"),
    ])
    st.markdown("&nbsp;")
    st.subheader("An advisor's book (illustrative synthetic clients)")
    book = table("demo_advisor_book.csv")
    st.dataframe(book, hide_index=True, use_container_width=True, column_config={
        "client": "Client", "tier": "Tier", "favourite_category": "Favourite category",
        "weeks_since_last_visit": st.column_config.NumberColumn("Weeks since last visit"),
        "spend_last_12m_eur": st.column_config.NumberColumn("Spend, 12 months", format="€%d"),
        "predicted_contact_value_eur": st.column_config.NumberColumn("Predicted contact value", format="€%d"),
        "champion_action": "Champion: best clients", "challenger_action": "Challenger: uplift",
    })
    provenance("Synthetic clients drawn from tier-level deciles; no real client appears. Consent is a hard gate "
               "and 'no action' is an explained outcome.")

    st.subheader("Product picks, judged on real purchases")
    rec = report("recommendations.json")
    allp = rec["test"]
    names = {"random": "Random", "only_chain_popularity": "Chain popularity", "only_visual_similarity":
             "Visual similarity alone", "blend_without_visual": "Blend without visual",
             "blend_with_visual": "Blend with visual"}
    left, right = st.columns(2)
    with left:
        values = [allp[k]["hit@5"] for k in names]
        chart(emphasis_bar(list(names.values()), values, {"Visual similarity alone", "Blend with visual"}, "{:.1%}",
                          "Whole collection: bought one of the top 5 picks"),
              pd.DataFrame({"method": list(names.values()), "hit@5": values}), key="recs_all")
    with right:
        new = rec["test_new_launches_only"]
        nl = {"random": "Random", "affinity_only": "Category/colour/fabric affinity", "visual_only":
              "Visual similarity alone", "affinity_plus_visual": "Affinity + visual"}
        values = [new[k]["hit@5"] for k in nl]
        chart(emphasis_bar(list(nl.values()), values, {"Affinity + visual"}, "{:.1%}",
                          "New launches (no sales yet): bought one of the top 5 picks"),
              pd.DataFrame({"method": list(nl.values()), "hit@5": values}), key="recs_new")
    gain = new["visual_gain_hit@5"]
    provenance(f"Image embeddings from the Silhouette Vision project. On new launches visual similarity adds "
               f"{gain['difference'] * 100:+.1f} points [{gain['ci_low'] * 100:.1f}, {gain['ci_high'] * 100:.1f}].")


def planning() -> None:
    header("Demand and allocation", "Where scarce pieces should go",
           "Demand recovered from stock-outs, a store-level launch forecast with uncertainty, and allocation "
           "judged on true demand for 890 launches (Jul–Nov 2019).")
    demand = report("demand.json")
    alloc = report("allocation.json")
    scarce = report("scarcity_scenario.json")
    left, right = st.columns(2)
    with left:
        sweep = pd.DataFrame(alloc["supply_sweep"])
        fig = go.Figure()
        for policy, colour, name in (("optimal_with_uncertainty", JADE, "Forecast-based (optimal)"),
                                     ("equal_split", GOLD, "Equal split"),
                                     ("retailer_opening_share", OXBLOOD, "Retailer's opening split"),
                                     ("oracle_true_demand", SAPPHIRE, "Oracle (true demand)")):
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
    models = demand["launch_forecast"]["models"]
    names = {"store_category_average": "Store × category average", "gbm|base": "Gradient boosting (chosen)",
             "gbm|base+image": "+ image features", "gbm|base+trends": "+ Google Trends",
             "gbm|base+image+trends": "+ both"}
    wape = [models[k]["test"]["wape"] for k in names]
    left, right = st.columns(2)
    with left:
        chart(emphasis_bar(list(names.values()), wape, {"Gradient boosting (chosen)", "Store × category average"},
                           "{:.3f}", "Launch forecast error on Jul–Nov 2019 launches (WAPE, lower is better)"),
              pd.DataFrame({"model": list(names.values()), "test WAPE": wape}), key="wape")
    with right:
        recov = demand["recovery"]["validation_artificial_censoring"]["censor_30pct_of_weeks"]
        methods = {"naive": "Sales = demand", "max": "Max(sales, expected)", "conditional": "Conditional (used)"}
        values = [recov[k]["total_bias_pct"] for k in methods]
        chart(emphasis_bar(list(methods.values()), values, {"Conditional (used)"}, "{:+.1f}%",
                           "Recovering demand in sold-out weeks: bias vs known truth", reference=(0, "unbiased")),
              pd.DataFrame({"method": list(methods.values()), "bias %": values}), key="recovery")
    st.subheader("Hero pieces: which clients get them")
    ca = report("client_allocation.json")
    policies = pd.DataFrame(ca["policies"]).T.reset_index(names="policy")
    st.dataframe(policies[["policy", "expected_revenue_eur_m", "expected_relationship_value_eur_m",
                           "expected_total_value_eur_m", "share_offers_to_vic", "share_offers_to_rising"]],
                 hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame(ca["relationship_value_sensitivity"]), hide_index=True, use_container_width=True)
    provenance("Integer programme per boutique: one hero piece per client, consent required, reseller-risk "
               "clients excluded, 30% reserved for rising clients. Acceptance and relationship value are "
               "assumptions; the sensitivity table shows how the choice moves with them.")


def tier_dial() -> None:
    header("Brand Tier Dial", "What it costs to stop discounting",
           "The same real demand under three brand policies. Markdown lift is measured on real data "
           "(staggered difference-in-differences); prices, margins and exit values are assumptions.")
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
    colours = {"Luxury": JADE, "Aspirational": GOLD, "Premium": SAPPHIRE, "Retailer's actual": OXBLOOD}
    policies["name"] = policies["policy"].map(names)
    fig = go.Figure(go.Bar(x=policies["name"], y=policies[column], marker_color=[colours[n] for n in policies["name"]],
                           text=[f"€{v:,.0f}M" for v in policies[column]], textposition="outside",
                           cliponaxis=False, textfont={"color": INK}))
    fig.update_layout(title=f"Gross profit after a {cost} brand cost per discounted unit ({scenario} lift)",
                      showlegend=False)
    fig.update_yaxes(range=padded(policies[column].tolist(), 0.12), title="€M")
    chart(fig, policies[["name", "units_bought", "full_price_share_of_units_sold", "sell_through", "revenue_eur_m",
                         "gross_profit_eur_m", column, "profit_margin"]])
    left, right = st.columns(2)
    with left:
        exits = pd.DataFrame(res["exit_value_sensitivity"])
        fig = go.Figure(go.Scatter(x=exits["exit_value_share_of_price"], y=exits["gross_profit_eur_m"],
                                   mode="lines+markers+text", line={"color": JADE, "width": 2}, marker={"size": 8},
                                   text=[f"€{v:,.0f}M" for v in exits["gross_profit_eur_m"]],
                                   textposition="top center", textfont={"color": INK}))
        fig.update_layout(title="Profit with no markdowns, by value recovered per unsold unit")
        y = exits["gross_profit_eur_m"]
        fig.update_yaxes(range=[y.min() - 0.15 * (y.max() - y.min()), y.max() + 0.2 * (y.max() - y.min())])
        fig.update_xaxes(tickformat=".0%", title="exit value, share of list price")
        chart(fig, exits[["exit_value_share_of_price", "units_bought", "gross_profit_eur_m", "profit_margin"]])
    with right:
        pulse = table("market_pulse.csv").dropna(subset=["change_pct"])
        pulse = pulse[pulse["term"] != "Louis Vuitton Speedy"]
        pulse = pulse[pulse["source_file"].isin(["aspirational_bags", "luxury_bags", "luxury_brands"])]
        pulse = pulse.sort_values("change_pct")
        fig = emphasis_bar(pulse["term"].tolist(), pulse["change_pct"].tolist(),
                           {"Coach Tabby", "Gucci", "Hermes Birkin"}, "{:+.0f}%",
                           "US search interest, last 12 months vs Oct 2021–Oct 2022", reference=(0, ""))
        fig.update_yaxes(autorange=True)
        chart(fig, pulse[["source_file", "term", "first_year_mean", "last_year_mean", "change_pct"]])
    effects = report("discount_effects.json")
    rows = pd.DataFrame(effects["markdown_lift"]["by_depth"])
    st.dataframe(rows[["depth", "events", "lift", "ci_low", "ci_high", "lift_weeks_3_4", "placebo_lift"]],
                 hide_index=True, use_container_width=True)
    provenance("Shallow markdowns pass the placebo check; deeper ones do not (the retailer discounts where sales "
               "are already slipping), hence the conservative scenario. Discount-acquired clients were not worse "
               "in this data, so the brand cost of discounting is an explicit assumption.")


def health() -> None:
    header("Model health", "Would we have seen it coming?",
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
    tiles([
        ("5 of 8", "inputs to act on between training and test", "explains the 2019 spend bias"),
        ("0 of 8", "inputs to act on before COVID", "the shock hit outcomes, not inputs"),
        (f"{cv['test']['lapse']['return_rate']:.1%} → {cv['covid']['lapse']['return_rate']:.1%}",
         "realised return rate, 2019 → 2020", "only outcome monitoring catches this"),
    ])
    st.markdown("&nbsp;")
    st.markdown("**Monitoring plan.** Track input PSI monthly; compare predicted with realised visits and spend "
                "every week; recalibrate spend totals on the latest completed year; fall back to "
                "last-year ranking when outcome bias passes 20%.")


def method() -> None:
    header("Method and gaps", "What is real, what is modelled, what is still open",
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


PAGES = [
    (executive, "Decision brief", ":material/insights:"),
    (clients, "Client intelligence", ":material/diamond:"),
    (clienteling, "Clienteling", ":material/support_agent:"),
    (planning, "Demand and allocation", ":material/inventory_2:"),
    (tier_dial, "Brand Tier Dial", ":material/tune:"),
    (health, "Model health", ":material/monitor_heart:"),
    (method, "Method and gaps", ":material/description:"),
]


def main() -> None:
    st.set_page_config(page_title="Couture Atelier Intelligence", page_icon="◆", layout="wide")
    style()
    pages = [st.Page(fn, title=title, icon=icon, default=(i == 0)) for i, (fn, title, icon) in enumerate(PAGES)]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
