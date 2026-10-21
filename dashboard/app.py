from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SYNTHETIC = ROOT / "data" / "synthetic"

st.set_page_config(page_title="Couture Atelier Intelligence", page_icon="CA", layout="wide")

COLORS = {
    "ink": "#18211F",
    "muted": "#66716D",
    "paper": "#F5F3EE",
    "panel": "#FBFAF7",
    "line": "#DCDDD5",
    "jade": "#1F5B52",
    "gold": "#B9975B",
    "coral": "#C8755A",
    "blue": "#527484",
}


def load(name: str, synthetic: bool = False) -> pd.DataFrame:
    folder = SYNTHETIC if synthetic else PROCESSED
    return pd.read_parquet(folder / f"{name}.parquet")


def load_json(name: str, synthetic: bool = False) -> dict:
    folder = SYNTHETIC if synthetic else PROCESSED
    return json.loads((folder / f"{name}.json").read_text())


@st.cache_data

def get_data() -> dict[str, pd.DataFrame | dict]:
    return {
        "customers": load("customer_360"),
        "segments": load("behavioral_segments"),
        "profiles": load("behavioral_segment_profiles"),
        "churn": load("churn_predictions"),
        "clv": load("client_clv"),
        "recommendations": load("clienteling_recommendations"),
        "daily": load("daily_sales"),
        "inventory": load("inventory_daily", synthetic=True),
        "transfers": load("transfer_recommendations", synthetic=True),
        "stores": load("stores", synthetic=True),
        "products": load("products", synthetic=True),
        "forecast_report": load_json("demand_model_report"),
        "inventory_report": load_json("inventory_optimization_report", synthetic=True),
        "client_report": load_json("client_behavior_report"),
        "clienteling_report": load_json("clienteling_report"),
        "baseline_report": load_json("baseline_report"),
    }


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');
        :root {{ --ink:{COLORS['ink']}; --muted:{COLORS['muted']}; --paper:{COLORS['paper']}; --panel:{COLORS['panel']}; --line:{COLORS['line']}; --jade:{COLORS['jade']}; --gold:{COLORS['gold']}; }}
        .stApp {{ background: var(--paper); color: var(--ink); }}
        [data-testid="stSidebar"] {{ background: var(--ink); border-right: 0; }}
        [data-testid="stSidebar"] * {{ color: #F5F3EE !important; }}
        .brand-mark {{ font-family:'Playfair Display', serif; font-size:29px; line-height:1.05; letter-spacing:0; color:#F5F3EE; margin:10px 0 4px; }}
        .brand-sub {{ color:#B9C5BF; font:500 10px 'DM Sans', sans-serif; letter-spacing:2px; text-transform:uppercase; }}
        .eyebrow {{ color:var(--jade); font:700 11px 'DM Sans',sans-serif; letter-spacing:2px; text-transform:uppercase; margin-bottom:7px; }}
        h1,h2,h3 {{ font-family:'Playfair Display',serif !important; font-weight:500 !important; letter-spacing:0 !important; color:var(--ink) !important; }}
        h1 {{ font-size:42px !important; line-height:1.08 !important; margin:0 0 5px !important; }}
        h2 {{ font-size:27px !important; margin:12px 0 8px !important; }}
        .subtitle {{ color:var(--muted); font:400 14px 'DM Sans',sans-serif; margin-bottom:26px; }}
        .kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:2px; padding:18px 20px 16px; min-height:112px; }}
        .kpi-label {{ color:var(--muted); font:600 10px 'DM Sans',sans-serif; letter-spacing:1.2px; text-transform:uppercase; }}
        .kpi-value {{ color:var(--ink); font:600 30px 'DM Sans',sans-serif; margin-top:10px; }}
        .kpi-note {{ color:var(--muted); font:400 11px 'DM Sans',sans-serif; margin-top:3px; }}
        .insight {{ border-left:3px solid var(--gold); background:#EFE9DC; padding:15px 18px; color:var(--ink); font:500 13px 'DM Sans',sans-serif; line-height:1.5; }}
        .section-rule {{ border-top:1px solid var(--line); margin:23px 0 16px; }}
        .stButton > button {{ border-radius:2px; border:1px solid #315F57; color:#F5F3EE; background:#1F5B52; font-weight:600; }}
        [data-testid="stMetric"] {{ background:var(--panel); border:1px solid var(--line); padding:13px 15px; }}
        [data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
        .caption {{ color:var(--muted); font:400 11px 'DM Sans',sans-serif; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans", "color": COLORS["ink"], "size": 11},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    fig.update_xaxes(showgrid=False, linecolor=COLORS["line"])
    fig.update_yaxes(showgrid=True, gridcolor="#E5E5DE", zeroline=False)
    return fig


def overview(data: dict) -> None:
    header("Couture Atelier / Intelligence Brief", "The house, in signal.", "A decision cockpit for demand, stock, client value, and discreet action.")
    rec = data["clienteling_report"]
    inv = data["inventory_report"]
    forecast = data["forecast_report"]["model"]
    baseline_forecast = data["baseline_report"]["forecast"]
    churn = data["client_report"]["churn"]
    cols = st.columns(4)
    with cols[0]: kpi("Client universe", f"{rec['customer_count']:,}", "synthetic client IDs")
    with cols[1]: kpi("Projected stockout", f"{inv['recommended_stockout_rate']:.1%}", f"{inv['stockout_rate_change']:+.1%} after transfers")
    forecast_delta = (forecast["mae_units"] / baseline_forecast["mae_units"] - 1) * 100
    with cols[2]: kpi("Forecast MAE", f"{forecast['mae_units']:.2f}", f"{forecast_delta:+.1f}% vs seasonal baseline")
    with cols[3]: kpi("Churn ROC-AUC", f"{churn['roc_auc']:.3f}", "chronological test cohort")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="eyebrow">Executive read</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight">The strongest signal is not volume. It is coordination: a demand model that informs inventory movement, while client value and consent determine whether a human relationship should be activated.</div>', unsafe_allow_html=True)
        st.markdown("### What deserves attention")
        attention = pd.DataFrame({"Signal": ["Demand model", "Inventory", "Clienteling"], "Read": ["MAE is 15.7% below the seasonal baseline.", "Transfers and warehouse replenishment reduce projected stockouts under explicit caps.", "1,176 clients correctly resolve to no action."]})
        st.dataframe(attention, hide_index=True, use_container_width=True, column_config={"Signal": st.column_config.TextColumn(width="small"), "Read": st.column_config.TextColumn(width="large")})
    with right:
        st.markdown('<div class="eyebrow">Action mix</div>', unsafe_allow_html=True)
        action_order = ["private_appointment", "re_engagement", "product_suggestion", "no_action"]
        actions = pd.DataFrame({"action": action_order, "clients": [rec["action_counts"].get(action, 0) for action in action_order]})
        fig = px.bar(actions, x="clients", y="action", orientation="h", color="action", color_discrete_sequence=[COLORS["jade"], COLORS["gold"], COLORS["coral"], COLORS["blue"]])
        fig.update_layout(showlegend=False, xaxis_title="Clients", yaxis_title="")
        st.plotly_chart(chart_layout(fig, 270), use_container_width=True)
    st.caption("All figures are derived or simulated portfolio outputs. They are not Louis Vuitton or private luxury-house data.")


def demand_inventory(data: dict) -> None:
    header("Decision 01 / Availability", "Protect desirability with precision.", "Forecast demand, see the pressure points, and move only what has a reason to move.")
    inv = data["inventory"]
    transfers = data["transfers"]
    products = data["products"]
    stores = data["stores"]
    cols = st.columns(4)
    with cols[0]: kpi("Hero products", f"{len(products):,}", "synthetic product universe")
    with cols[1]: kpi("Boutiques", f"{len(stores):,}", "six operating formats")
    with cols[2]: kpi("Transfer units", f"{int(transfers['transfer_units'].sum()):,}", "recommended next movement")
    with cols[3]: kpi("Stockout rows", f"{int(inv['stockout_flag'].sum()):,}", "90-day synthetic history")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="eyebrow">Transfer ledger</div>', unsafe_allow_html=True)
        view = transfers.merge(products[["product_id", "category", "price_tier"]], on="product_id", how="left")
        st.dataframe(view[["product_id", "category", "from_store_id", "to_store_id", "transfer_units", "reason"]], hide_index=True, use_container_width=True, height=335)
    with right:
        st.markdown('<div class="eyebrow">Stockout pressure by store</div>', unsafe_allow_html=True)
        store_rate = inv.groupby("store_id", as_index=False).agg(stockout_rate=("stockout_flag", "mean")).sort_values("stockout_rate")
        fig = px.bar(store_rate, x="stockout_rate", y="store_id", orientation="h", color="stockout_rate", color_continuous_scale=["#D8E5DE", COLORS["jade"]], labels={"stockout_rate": "Stockout rate", "store_id": "Store"})
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(chart_layout(fig, 335), use_container_width=True)
    st.markdown('<div class="insight">The current policy respects a minimum display unit, caps donor transfers, and uses a finite central-warehouse capacity for residual deficits. Transfer cost and lead time remain the next operational refinements.</div>', unsafe_allow_html=True)


def client_intelligence(data: dict) -> None:
    header("Decision 02 / Client value", "Know the relationship before you touch it.", "A quiet view of value, risk, and behavior designed for human judgment.")
    clv = data["clv"].merge(data["churn"][["customer_id", "churn_probability", "risk_band"]], on="customer_id", how="left")
    segments = data["segments"].merge(data["clv"][["customer_id", "predicted_clv_90d"]], on="customer_id", how="left")
    cols = st.columns(4)
    with cols[0]: kpi("90-day predicted value", f"{clv['predicted_clv_90d'].sum():,.0f}", "derived CLV total")
    with cols[1]: kpi("High-risk clients", f"{int(clv['risk_band'].eq('high').sum()):,}", "churn probability >= 0.66")
    with cols[2]: kpi("Behavioral segments", f"{segments['behavior_segment'].nunique():,}", "deterministic K-means")
    with cols[3]: kpi("CLV correlation", "0.686", "next-90-day realization")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown('<div class="eyebrow">Value / risk map</div>', unsafe_allow_html=True)
        fig = px.scatter(clv.sample(min(3000, len(clv)), random_state=42), x="predicted_clv_90d", y="churn_probability", color="risk_band", color_discrete_map={"low": COLORS["jade"], "medium": COLORS["gold"], "high": COLORS["coral"]}, opacity=0.6, labels={"predicted_clv_90d": "Predicted 90-day value", "churn_probability": "Churn probability"})
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    with right:
        st.markdown('<div class="eyebrow">Behavioral profiles</div>', unsafe_allow_html=True)
        profiles = data["profiles"].copy()
        profiles["behavior_segment"] = profiles["behavior_segment"].astype(str)
        fig = px.bar(profiles, x="behavior_segment", y="historical_revenue", color="customer_count", color_continuous_scale=["#D8E5DE", COLORS["jade"]], labels={"historical_revenue": "Average historical revenue", "behavior_segment": "Segment"})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    st.caption("Segments are analytical labels, not identities or inferred wealth categories.")


def clienteling(data: dict) -> None:
    header("Decision 03 / Clienteling", "Make relevance feel effortless.", "A prioritized associate workspace with consent, availability, and restraint built in.")
    recs = data["recommendations"].copy()
    cols = st.columns(4)
    with cols[0]: kpi("Consented", f"{int(recs['consent_status'].eq('consented').sum()):,}", "eligible relationship base")
    with cols[1]: kpi("No action", f"{int(recs['action_type'].eq('no_action').sum()):,}", "restraint is a valid result")
    with cols[2]: kpi("Re-engagement", f"{int(recs['action_type'].eq('re_engagement').sum()):,}", "elevated churn risk")
    with cols[3]: kpi("Appointments", f"{int(recs['action_type'].eq('private_appointment').sum()):,}", "high-value clients")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    action = st.selectbox("Decision view", ["All decisions", "Eligible actions", "Re-engagement", "Private appointment", "Product suggestion", "No action"])
    labels = {"Re-engagement": "re_engagement", "Private appointment": "private_appointment", "Product suggestion": "product_suggestion", "No action": "no_action"}
    if action == "All decisions":
        selected = recs
    elif action == "Eligible actions":
        selected = recs[recs["eligible_for_clienteling"]]
    else:
        selected = recs[recs["action_type"] == labels[action]]
    selected = selected.sort_values(["churn_probability", "predicted_clv_90d"], ascending=[False, False])
    st.dataframe(selected[["customer_id", "action_type", "recommended_product_id", "predicted_clv_90d", "churn_probability", "reason"]].head(100), hide_index=True, use_container_width=True, height=405)
    st.caption(f"Showing {len(selected):,} decisions. Sort priority: churn risk, then predicted 90-day value.")
    st.caption("This is associate decision support, not automatic outreach. Never contact a client without consent.")


def main() -> None:
    inject_css()
    data = get_data()
    with st.sidebar:
        st.markdown('<div class="brand-mark">Couture<br>Atelier</div><div class="brand-sub">Intelligence studio</div>', unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("Navigate", ["Executive pulse", "Demand & inventory", "Client intelligence", "Clienteling"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown('<div class="brand-sub">PORTFOLIO SIMULATION</div>', unsafe_allow_html=True)
        st.caption("Public observations + synthetic luxury layer")
    {"Executive pulse": overview, "Demand & inventory": demand_inventory, "Client intelligence": client_intelligence, "Clienteling": clienteling}[page](data)


if __name__ == "__main__":
    main()
