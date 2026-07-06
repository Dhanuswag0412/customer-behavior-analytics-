"""
=============================================================================
 ChurnIQ — Customer Behavior Analytics Platform
 Project : Customer Behavior Analytics Using Clustering and Predictive
           Modelling

 A premium, industry-grade Streamlit dashboard that presents:
   - Executive KPI overview
   - Exploratory Data Analysis (EDA)
   - Customer Segmentation (K-Means personas)
   - Predictive Modelling (Random Forest churn prediction + live scoring)
   - Feature Importance
   - Customer Insights & Recommendations
   - Reports & Export

 Run:  streamlit run app.py
=============================================================================
"""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Keep string columns as classic numpy object dtype (not pandas' newer
# StringDtype/ArrowStringArray) so data loaded here behaves the same way
# regardless of which pandas version is installed.
try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnIQ | Customer Behavior Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "processed_customer_data.csv"
MODEL_PATH = "customer_churn_model.pkl"

# --------------------------------------------------------------------------
# THEME TOKENS
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#0B1220",
    "bg_alt": "#0F1830",
    "surface": "#141F38",
    "surface_alt": "#1A2745",
    "border": "#25335A",
    "text": "#EDF1FA",
    "text_dim": "#93A0C2",
    "primary": "#6C5CE7",
    "primary_dim": "#8B7FEF",
    "teal": "#17C3B2",
    "amber": "#F4A527",
    "red": "#F5556C",
    "gradient": "linear-gradient(135deg, #6C5CE7 0%, #17C3B2 100%)",
}

PERSONA_COLORS = {
    "Champions": "#17C3B2",
    "Loyal Customers": "#6C5CE7",
    "At-Risk Customers": "#F4A527",
    "Dormant / Low-Engagement": "#F5556C",
}

# --------------------------------------------------------------------------
# GLOBAL CSS
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 15% 0%, #131E3D 0%, {COLORS['bg']} 45%) fixed;
    color: {COLORS['text']};
}}

/* Hide default streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
div.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1730 0%, #0A1226 100%);
    border-right: 1px solid {COLORS['border']};
}}
section[data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}

/* Headings */
h1, h2, h3, h4 {{
    font-family: 'Poppins', sans-serif !important;
    color: {COLORS['text']} !important;
    letter-spacing: -0.01em;
}}

/* Hero banner */
.hero {{
    background: {COLORS['gradient']};
    border-radius: 20px;
    padding: 34px 40px;
    margin-bottom: 26px;
    box-shadow: 0 20px 45px -18px rgba(108,92,231,0.55);
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.6s ease;
}}
.hero::after {{
    content:"";
    position:absolute; top:-60%; right:-10%;
    width:380px; height:380px; border-radius:50%;
    background: rgba(255,255,255,0.10);
}}
.hero h1 {{
    color:#fff !important; font-size: 32px; margin:0 0 6px 0; font-weight:800;
}}
.hero p {{ color: rgba(255,255,255,0.9); margin:0; font-size:15px; }}
.hero .badge {{
    display:inline-block; background: rgba(255,255,255,0.18); color:#fff;
    padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600;
    margin-bottom:14px; letter-spacing:0.03em;
}}

/* KPI cards */
.kpi-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 16px;
    padding: 20px 22px;
    height: 100%;
    transition: transform .18s ease, box-shadow .18s ease;
    animation: fadeIn 0.5s ease;
}}
.kpi-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 14px 30px -12px rgba(108,92,231,0.35);
    border-color: {COLORS['primary_dim']};
}}
.kpi-label {{
    color:{COLORS['text_dim']}; font-size:12.5px; font-weight:600;
    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;
}}
.kpi-value {{ font-family:'Poppins',sans-serif; font-size:28px; font-weight:700; color:{COLORS['text']}; }}
.kpi-delta {{ font-size:12.5px; font-weight:600; margin-top:6px; }}
.kpi-delta.pos {{ color:{COLORS['teal']}; }}
.kpi-delta.neg {{ color:{COLORS['red']}; }}
.kpi-icon {{ font-size:22px; margin-bottom:6px; }}

/* Section header */
.section-header {{
    font-family:'Poppins',sans-serif; font-weight:700; font-size:20px;
    margin: 26px 0 12px 0; color:{COLORS['text']};
    display:flex; align-items:center; gap:10px;
}}
.section-sub {{ color:{COLORS['text_dim']}; font-size:13.5px; margin-top:-8px; margin-bottom:16px; }}

/* Generic card / persona card */
.card {{
    background:{COLORS['surface']}; border:1px solid {COLORS['border']};
    border-radius:16px; padding:20px; animation: fadeIn .5s ease;
}}
.persona-card {{
    background:{COLORS['surface']}; border:1px solid {COLORS['border']};
    border-left: 5px solid var(--accent, {COLORS['primary']});
    border-radius:14px; padding:18px 20px; margin-bottom:14px;
    transition: transform .18s ease;
}}
.persona-card:hover {{ transform: translateX(3px); }}
.persona-title {{ font-family:'Poppins',sans-serif; font-weight:700; font-size:16.5px; margin-bottom:4px;}}
.persona-desc {{ color:{COLORS['text_dim']}; font-size:13px; line-height:1.5; }}
.pill {{
    display:inline-block; padding:3px 10px; border-radius:20px; font-size:11.5px;
    font-weight:700; letter-spacing:.02em;
}}

/* Insight bullet card */
.insight-card {{
    background:{COLORS['surface_alt']}; border:1px solid {COLORS['border']};
    border-radius:14px; padding:16px 18px; margin-bottom:12px; font-size:14px;
    line-height:1.55; border-left:4px solid {COLORS['teal']};
}}
.reco-card {{
    background:{COLORS['surface']}; border:1px solid {COLORS['border']};
    border-radius:14px; padding:18px; margin-bottom:12px;
}}

/* Metric row divider */
hr.soft {{ border: none; border-top: 1px solid {COLORS['border']}; margin: 18px 0; }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
    background: {COLORS['gradient']}; color:#fff; border:none; border-radius:10px;
    padding:0.55rem 1.1rem; font-weight:600; transition: all .15s ease;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    filter: brightness(1.08); transform: translateY(-1px);
}}

/* Dataframe */
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}

/* Animations */
@keyframes fadeIn {{ from {{opacity:0; transform: translateY(6px);}} to {{opacity:1; transform: translateY(0);}} }}

/* Radio-as-nav fallback styling */
div[role="radiogroup"] label {{
    background:{COLORS['surface']}; border:1px solid {COLORS['border']};
    padding:8px 12px; border-radius:10px; margin-bottom:6px; width:100%;
}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# DATA / MODEL LOADING
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading customer data ...")
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Loading trained models ...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


df = load_data()
bundle = load_model()

PERSONAS = bundle["cluster_profile"].sort_values("Cluster")["Persona"].tolist()


# --------------------------------------------------------------------------
# SMALL HELPERS
# --------------------------------------------------------------------------
def kpi_card(label, value, icon="📊", delta=None, delta_positive=True):
    delta_html = ""
    if delta is not None:
        cls = "pos" if delta_positive else "neg"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle=None, icon=""):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def style_fig(fig, height=380):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    return fig


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
PAGES = [
    ("Overview", "speedometer2"),
    ("Exploratory Data Analysis", "bar-chart-line"),
    ("Customer Segmentation", "diagram-3"),
    ("Predictive Modelling", "cpu"),
    ("Feature Importance", "sort-down"),
    ("Insights & Recommendations", "lightbulb"),
    ("Reports & Export", "file-earmark-arrow-down"),
]
PAGE_ICONS_EMOJI = ["🏠", "📊", "🧩", "🤖", "🔥", "💡", "📥"]

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 6px 0 18px 0;">
        <div style="font-size:34px;">🧠</div>
        <div style="font-family:'Poppins',sans-serif; font-weight:800; font-size:19px; color:{COLORS['text']};">ChurnIQ</div>
        <div style="font-size:11.5px; color:{COLORS['text_dim']}; letter-spacing:0.04em;">CUSTOMER INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    if HAS_OPTION_MENU:
        selected = option_menu(
            menu_title=None,
            options=[p[0] for p in PAGES],
            icons=[p[1] for p in PAGES],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": COLORS["primary_dim"], "font-size": "15px"},
                "nav-link": {
                    "font-size": "14px", "text-align": "left", "margin": "3px 0",
                    "border-radius": "10px", "color": COLORS["text"],
                    "padding": "10px 12px",
                },
                "nav-link-selected": {"background-color": COLORS["primary"], "color": "#fff"},
            },
        )
    else:
        labels = [f"{e}  {p[0]}" for e, p in zip(PAGE_ICONS_EMOJI, PAGES)]
        choice = st.radio("Navigate", labels, label_visibility="collapsed")
        selected = PAGES[labels.index(choice)][0]

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:12px; color:{COLORS['text_dim']}; line-height:1.6;">
        <b style="color:{COLORS['text']};">Dataset</b><br>
        {len(df):,} customers · {df['Country'].nunique()} countries<br><br>
        <b style="color:{COLORS['text']};">Model</b><br>
        Random Forest · K-Means (k={bundle['n_clusters']})<br><br>
        <b style="color:{COLORS['text']};">Last refreshed</b><br>
        {datetime.now().strftime('%d %b %Y')}
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------------------------------
def page_overview():
    st.markdown(f"""
    <div class="hero">
        <div class="badge">● LIVE ANALYTICS</div>
        <h1>Customer Behavior Analytics</h1>
        <p>Clustering &amp; predictive modelling platform turning raw e-commerce behavior into
        retention-ready decisions — segment customers, predict churn, and act on it.</p>
    </div>
    """, unsafe_allow_html=True)

    total_customers = len(df)
    churn_rate = df["Churned"].mean() * 100
    avg_ltv = df["Lifetime_Value"].mean()
    avg_aov = df["Average_Order_Value"].mean()
    n_segments = df["Persona"].nunique()
    accuracy = bundle["metrics"]["accuracy"] * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Total Customers", f"{total_customers:,}", "👥")
    with c2: kpi_card("Churn Rate", f"{churn_rate:.1f}%", "⚠️", f"{(1-churn_rate/100)*100:.1f}% retained", False)
    with c3: kpi_card("Avg Lifetime Value", f"${avg_ltv:,.0f}", "💎")
    with c4: kpi_card("Avg Order Value", f"${avg_aov:,.0f}", "🛒")
    with c5: kpi_card("Customer Segments", f"{n_segments}", "🧩")
    with c6: kpi_card("Model Accuracy", f"{accuracy:.1f}%", "🎯", "ROC-AUC "+f"{bundle['metrics']['roc_auc']:.2f}", True)

    section_header("Churn Overview & Segment Health", "How retention and risk are distributed across the customer base", "📈")
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        churn_counts = df["Churned"].value_counts().rename({0: "Retained", 1: "Churned"})
        fig = go.Figure(go.Pie(
            labels=churn_counts.index, values=churn_counts.values, hole=0.62,
            marker=dict(colors=[COLORS["teal"], COLORS["red"]]),
            textinfo="percent", textfont=dict(color="#fff", size=13),
        ))
        fig.update_layout(title="Retention vs Churn", showlegend=True)
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)

    with col2:
        seg_counts = df["Persona"].value_counts()
        fig = go.Figure(go.Pie(
            labels=seg_counts.index, values=seg_counts.values, hole=0.62,
            marker=dict(colors=[PERSONA_COLORS.get(p, COLORS["primary"]) for p in seg_counts.index]),
            textinfo="percent", textfont=dict(color="#fff", size=12),
        ))
        fig.update_layout(title="Customer Segments")
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)

    with col3:
        top_countries = df["Country"].value_counts().head(8).sort_values()
        fig = go.Figure(go.Bar(
            x=top_countries.values, y=top_countries.index, orientation="h",
            marker=dict(color=top_countries.values, colorscale=[[0, COLORS["primary"]], [1, COLORS["teal"]]]),
        ))
        fig.update_layout(title="Customers by Country")
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)

    section_header("Acquisition & Churn Trend by Signup Quarter", "Pseudo-time trend using signup cohort as a proxy for acquisition trend", "🗓️")
    col1, col2 = st.columns(2)
    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    with col1:
        q_counts = df["Signup_Quarter"].value_counts().reindex(quarter_order)
        fig = go.Figure(go.Bar(x=q_counts.index, y=q_counts.values,
                                marker=dict(color=COLORS["primary"])))
        fig.update_layout(title="New Customers Acquired per Quarter")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with col2:
        q_churn = df.groupby("Signup_Quarter")["Churned"].mean().reindex(quarter_order) * 100
        fig = go.Figure(go.Scatter(x=q_churn.index, y=q_churn.values, mode="lines+markers",
                                    line=dict(color=COLORS["red"], width=3), marker=dict(size=9)))
        fig.update_layout(title="Churn Rate (%) by Signup Quarter")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    section_header("Segment Snapshot", "Quick view of each behavioral persona — full profiles on the Segmentation page", "🧬")
    cols = st.columns(4)
    profile = bundle["cluster_profile"].sort_values("Churn_Rate_%")
    for col, (_, row) in zip(cols, profile.iterrows()):
        accent = PERSONA_COLORS.get(row["Persona"], COLORS["primary"])
        with col:
            st.markdown(f"""
            <div class="persona-card" style="--accent:{accent};">
                <div class="persona-title">{row['Persona']}</div>
                <div class="persona-desc">{row['Size']:,} customers · {row['Churn_Rate_%']:.1f}% churn<br>
                Avg LTV ${row['Lifetime_Value']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE 2: EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------------
def page_eda():
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Filter the customer base and explore behavioral distributions, correlations, and churn drivers.</div>', unsafe_allow_html=True)

    with st.expander("🔎 Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        countries = f1.multiselect("Country", sorted(df["Country"].unique()), default=[])
        genders = f2.multiselect("Gender", sorted(df["Gender"].unique()), default=[])
        churn_filter = f3.selectbox("Churn Status", ["All", "Retained", "Churned"])

    fdf = df.copy()
    if countries:
        fdf = fdf[fdf["Country"].isin(countries)]
    if genders:
        fdf = fdf[fdf["Gender"].isin(genders)]
    if churn_filter == "Retained":
        fdf = fdf[fdf["Churned"] == 0]
    elif churn_filter == "Churned":
        fdf = fdf[fdf["Churned"] == 1]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Filtered Customers", f"{len(fdf):,}", "👥")
    with c2: kpi_card("Churn Rate", f"{fdf['Churned'].mean()*100:.1f}%", "⚠️")
    with c3: kpi_card("Avg Session (min)", f"{fdf['Session_Duration_Avg'].mean():.1f}", "⏱️")
    with c4: kpi_card("Avg Cart Abandonment", f"{fdf['Cart_Abandonment_Rate'].mean():.1f}%", "🛍️")

    numeric_cols = [c for c in bundle["numerical_cols"] if c in fdf.columns]

    section_header("Feature Distribution Explorer", "Select any behavioral metric to inspect its distribution split by churn outcome", "📐")
    feat = st.selectbox("Feature", numeric_cols, index=numeric_cols.index("Lifetime_Value") if "Lifetime_Value" in numeric_cols else 0)
    col1, col2 = st.columns([1.4, 1])
    with col1:
        fig = px.histogram(
            fdf, x=feat, color=fdf["Churned"].map({0: "Retained", 1: "Churned"}),
            barmode="overlay", nbins=40, opacity=0.75,
            color_discrete_map={"Retained": COLORS["teal"], "Churned": COLORS["red"]},
        )
        fig.update_layout(title=f"Distribution of {feat}", legend_title_text="")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with col2:
        fig = px.box(
            fdf, x=fdf["Churned"].map({0: "Retained", 1: "Churned"}), y=feat,
            color=fdf["Churned"].map({0: "Retained", 1: "Churned"}),
            color_discrete_map={"Retained": COLORS["teal"], "Churned": COLORS["red"]},
        )
        fig.update_layout(title=f"{feat} by Outcome", showlegend=False, xaxis_title="")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    section_header("Correlation Heatmap", "Linear relationships across behavioral & financial metrics", "🔗")
    corr = fdf[numeric_cols + ["Churned"]].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, COLORS["red"]], [0.5, COLORS["surface_alt"]], [1, COLORS["teal"]]],
        zmid=0, colorbar=dict(thickness=14),
    ))
    fig.update_layout(title="Feature Correlation Matrix")
    st.plotly_chart(style_fig(fig, 620), use_container_width=True)

    section_header("Churn Rate by Category", "Where churn concentrates across demographics and acquisition cohorts", "🧭")
    col1, col2, col3 = st.columns(3)
    with col1:
        g = fdf.groupby("Country")["Churned"].mean().sort_values(ascending=False) * 100
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker=dict(color=COLORS["primary"])))
        fig.update_layout(title="Churn Rate (%) by Country")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with col2:
        g = fdf.groupby("Gender")["Churned"].mean().sort_values(ascending=False) * 100
        fig = go.Figure(go.Bar(x=g.index, y=g.values, marker=dict(color=COLORS["teal"])))
        fig.update_layout(title="Churn Rate (%) by Gender")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with col3:
        g = fdf.groupby("Signup_Quarter")["Churned"].mean().reindex(["Q1", "Q2", "Q3", "Q4"]) * 100
        fig = go.Figure(go.Bar(x=g.index, y=g.values, marker=dict(color=COLORS["amber"])))
        fig.update_layout(title="Churn Rate (%) by Signup Quarter")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    with st.expander("📋 View Filtered Raw Data"):
        st.dataframe(fdf.head(500), use_container_width=True, height=320)


# --------------------------------------------------------------------------
# PAGE 3: CUSTOMER SEGMENTATION
# --------------------------------------------------------------------------
def page_segmentation():
    section_header("Customer Segmentation", "Unsupervised K-Means clustering groups customers into behavioral personas", "🧩")

    profile = bundle["cluster_profile"].sort_values("Cluster")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Segments Found", f"{bundle['n_clusters']}", "🧩")
    with c2: kpi_card("Silhouette Score", f"{bundle['silhouette_score']:.3f}", "📐")
    with c3: kpi_card("Largest Segment", profile.loc[profile['Size'].idxmax(), 'Persona'], "👑")
    with c4: kpi_card("Highest Risk Segment", profile.loc[profile['Churn_Rate_%'].idxmax(), 'Persona'], "🚨")

    section_header("Customer Personas", "Auto-generated from cluster centroids using lifetime value, engagement, recency and churn signal", "🪪")
    cols = st.columns(2)
    for i, (_, row) in enumerate(profile.iterrows()):
        accent = PERSONA_COLORS.get(row["Persona"], COLORS["primary"])
        with cols[i % 2]:
            st.markdown(f"""
            <div class="persona-card" style="--accent:{accent};">
                <span class="pill" style="background:{accent}22; color:{accent};">{row['Size']:,} customers</span>
                <span class="pill" style="background:{COLORS['surface_alt']}; color:{COLORS['text_dim']}; margin-left:6px;">{row['Churn_Rate_%']:.1f}% churn</span>
                <div class="persona-title" style="margin-top:10px;">{row['Persona']}</div>
                <div class="persona-desc">{row['Description']}</div>
                <div class="persona-desc" style="margin-top:8px;">
                    💎 Avg LTV: <b style="color:{COLORS['text']};">${row['Lifetime_Value']:,.0f}</b> &nbsp;|&nbsp;
                    🛒 Avg Order: <b style="color:{COLORS['text']};">${row['Average_Order_Value']:,.0f}</b> &nbsp;|&nbsp;
                    📅 Days Since Purchase: <b style="color:{COLORS['text']};">{row['Days_Since_Last_Purchase']:.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    section_header("Segment Visualization", "2D projection (PCA) of the customer base colored by assigned persona", "🗺️")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        sample = df.sample(min(6000, len(df)), random_state=42)
        fig = px.scatter(
            sample, x="PCA1", y="PCA2", color="Persona",
            color_discrete_map=PERSONA_COLORS, opacity=0.55,
            hover_data=["Lifetime_Value", "Total_Purchases", "Churned"],
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(title="Customer Segments in 2D Space (PCA)")
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    with col2:
        radar_cols = ["Lifetime_Value", "Total_Purchases", "Login_Frequency",
                      "Average_Order_Value", "Cart_Abandonment_Rate"]
        norm = profile.set_index("Persona")[radar_cols]
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
        fig = go.Figure()
        for persona in norm.index:
            fig.add_trace(go.Scatterpolar(
                r=norm.loc[persona].values.tolist() + [norm.loc[persona].values[0]],
                theta=radar_cols + [radar_cols[0]],
                fill="toself", name=persona,
                line=dict(color=PERSONA_COLORS.get(persona, COLORS["primary"])),
                opacity=0.55,
            ))
        fig.update_layout(
            title="Segment Behavior Radar (normalized)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(showticklabels=False, gridcolor=COLORS["border"]),
                angularaxis=dict(gridcolor=COLORS["border"]),
            ),
        )
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    section_header("Segment Size & Value Distribution", "Where the customer base sits by size vs. average lifetime value", "📦")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=profile["Persona"], y=profile["Size"],
            marker=dict(color=[PERSONA_COLORS.get(p, COLORS["primary"]) for p in profile["Persona"]]),
        ))
        fig.update_layout(title="Customers per Segment")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with col2:
        fig = go.Figure(go.Bar(
            x=profile["Persona"], y=profile["Lifetime_Value"],
            marker=dict(color=[PERSONA_COLORS.get(p, COLORS["primary"]) for p in profile["Persona"]]),
        ))
        fig.update_layout(title="Average Lifetime Value per Segment")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)


# --------------------------------------------------------------------------
# PAGE 4: PREDICTIVE MODELLING
# --------------------------------------------------------------------------
def predict_churn(inputs: dict):
    """Run a single customer's raw inputs through the full saved pipeline:
    label-encode -> cluster scale -> KMeans -> assemble RF features -> scale -> predict."""
    le_dict = bundle["label_encoders"]
    row = {}
    for col in bundle["cluster_feature_cols"]:
        if col in bundle["categorical_cols"]:
            le = le_dict[col]
            val = inputs[col]
            if val not in le.classes_:
                val = le.classes_[0]
            row[col] = le.transform([val])[0]
        else:
            row[col] = inputs[col]

    cluster_vec = pd.DataFrame([row])[bundle["cluster_feature_cols"]]
    cluster_scaled = bundle["scaler_cluster"].transform(cluster_vec)
    cluster_id = int(bundle["kmeans_model"].predict(cluster_scaled)[0])
    persona = bundle["cluster_to_persona"][cluster_id]

    row["Cluster"] = cluster_id
    rf_vec = pd.DataFrame([row])[bundle["feature_cols_rf"]]
    rf_scaled = bundle["scaler_rf"].transform(rf_vec)
    proba = bundle["rf_model"].predict_proba(rf_scaled)[0, 1]
    pred = int(proba >= 0.5)
    return proba, pred, cluster_id, persona


def page_predictive():
    section_header("Predictive Modelling", "Random Forest classifier trained to flag customers likely to churn", "🤖")

    m = bundle["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Accuracy", f"{m['accuracy']*100:.1f}%", "🎯")
    with c2: kpi_card("Precision", f"{m['precision']*100:.1f}%", "🔍")
    with c3: kpi_card("Recall", f"{m['recall']*100:.1f}%", "📡")
    with c4: kpi_card("F1 Score", f"{m['f1_score']*100:.1f}%", "⚖️")
    with c5: kpi_card("ROC-AUC", f"{m['roc_auc']:.3f}", "📈")

    section_header("Model Diagnostics", "Confusion matrix and ROC curve on the held-out 20% test set", "🩺")
    col1, col2 = st.columns(2)
    with col1:
        cm = bundle["confusion_matrix"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: Retained", "Predicted: Churned"],
            y=["Actual: Retained", "Actual: Churned"],
            colorscale=[[0, COLORS["surface_alt"]], [1, COLORS["primary"]]],
            text=cm, texttemplate="%{text}", textfont=dict(size=18, color="#fff"),
            showscale=False,
        ))
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with col2:
        fpr, tpr = bundle["roc_curve"]["fpr"], bundle["roc_curve"]["tpr"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={m['roc_auc']:.3f})",
                                  line=dict(color=COLORS["teal"], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
                                  line=dict(color=COLORS["text_dim"], width=1.5, dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    with st.expander("📋 Full Classification Report"):
        rep = pd.DataFrame(bundle["classification_report"]).T.round(3)
        st.dataframe(rep, use_container_width=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_header("Live Churn Prediction", "Score a hypothetical or real customer instantly using the trained pipeline", "⚡")

    with st.form("predict_form"):
        st.markdown("**Demographics & Membership**")
        f1, f2, f3, f4 = st.columns(4)
        age = f1.slider("Age", 18, 80, 35)
        gender = f2.selectbox("Gender", sorted(df["Gender"].unique()))
        country = f3.selectbox("Country", sorted(df["Country"].unique()))
        city = f4.selectbox("City", sorted(df["City"].unique()))

        f5, f6, f7, f8 = st.columns(4)
        membership_years = f5.slider("Membership Years", 0.0, 10.0, 2.5)
        signup_quarter = f6.selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])
        payment_diversity = f7.slider("Payment Method Diversity", 1.0, 5.0, 2.0)
        credit_balance = f8.slider("Credit Balance ($)", 0.0, 5000.0, 1500.0)

        st.markdown("**Engagement Behavior**")
        g1, g2, g3, g4 = st.columns(4)
        login_freq = g1.slider("Login Frequency (per month)", 0.0, 30.0, 12.0)
        session_dur = g2.slider("Avg Session Duration (min)", 1.0, 60.0, 25.0)
        pages_session = g3.slider("Pages per Session", 1.0, 20.0, 6.0)
        email_open = g4.slider("Email Open Rate (%)", 0.0, 100.0, 30.0)

        g5, g6, g7 = st.columns(3)
        social_engagement = g5.slider("Social Media Engagement Score", 0.0, 100.0, 25.0)
        mobile_usage = g6.slider("Mobile App Usage (sessions/mo)", 0.0, 40.0, 15.0)
        cs_calls = g7.slider("Customer Service Calls", 0.0, 15.0, 2.0)

        st.markdown("**Purchase Behavior**")
        h1, h2, h3, h4 = st.columns(4)
        total_purchases = h1.slider("Total Purchases", 0.0, 60.0, 10.0)
        aov = h2.slider("Average Order Value ($)", 10.0, 500.0, 100.0)
        days_since = h3.slider("Days Since Last Purchase", 0.0, 180.0, 30.0)
        cart_abandon = h4.slider("Cart Abandonment Rate (%)", 0.0, 100.0, 40.0)

        h5, h6, h7, h8 = st.columns(4)
        discount_usage = h5.slider("Discount Usage Rate (%)", 0.0, 100.0, 20.0)
        returns_rate = h6.slider("Returns Rate (%)", 0.0, 30.0, 5.0)
        wishlist = h7.slider("Wishlist Items", 0.0, 20.0, 3.0)
        reviews = h8.slider("Product Reviews Written", 0.0, 20.0, 2.0)

        ltv = st.slider("Lifetime Value ($)", 0.0, 10000.0, 1200.0)

        submitted = st.form_submit_button("🔮 Predict Churn Risk")

    if submitted:
        inputs = {
            "Age": age, "Gender": gender, "Country": country, "City": city,
            "Membership_Years": membership_years, "Login_Frequency": login_freq,
            "Session_Duration_Avg": session_dur, "Pages_Per_Session": pages_session,
            "Cart_Abandonment_Rate": cart_abandon, "Wishlist_Items": wishlist,
            "Total_Purchases": total_purchases, "Average_Order_Value": aov,
            "Days_Since_Last_Purchase": days_since, "Discount_Usage_Rate": discount_usage,
            "Returns_Rate": returns_rate, "Email_Open_Rate": email_open,
            "Customer_Service_Calls": cs_calls, "Product_Reviews_Written": reviews,
            "Social_Media_Engagement_Score": social_engagement, "Mobile_App_Usage": mobile_usage,
            "Payment_Method_Diversity": payment_diversity, "Lifetime_Value": ltv,
            "Credit_Balance": credit_balance, "Signup_Quarter": signup_quarter,
        }
        proba, pred, cluster_id, persona = predict_churn(inputs)
        risk_label = "High" if proba >= 0.6 else ("Medium" if proba >= 0.3 else "Low")
        risk_color = COLORS["red"] if risk_label == "High" else (COLORS["amber"] if risk_label == "Medium" else COLORS["teal"])

        col1, col2 = st.columns([1, 1.3])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"color": COLORS["text"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLORS["text_dim"]},
                    "bar": {"color": risk_color},
                    "bgcolor": COLORS["surface"],
                    "steps": [
                        {"range": [0, 30], "color": "rgba(23,195,178,0.25)"},
                        {"range": [30, 60], "color": "rgba(244,165,39,0.25)"},
                        {"range": [60, 100], "color": "rgba(245,85,108,0.25)"},
                    ],
                },
                title={"text": "Churn Probability", "font": {"color": COLORS["text"]}},
            ))
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        with col2:
            st.markdown(f"""
            <div class="card">
                <span class="pill" style="background:{risk_color}22; color:{risk_color};">{risk_label} Risk</span>
                <span class="pill" style="background:{PERSONA_COLORS.get(persona, COLORS['primary'])}22; color:{PERSONA_COLORS.get(persona, COLORS['primary'])}; margin-left:6px;">{persona}</span>
                <div style="margin-top:14px; font-size:14px; color:{COLORS['text_dim']}; line-height:1.6;">
                    Prediction: <b style="color:{COLORS['text']};">{"Likely to churn" if pred==1 else "Likely to stay"}</b><br>
                    Assigned segment: <b style="color:{COLORS['text']};">{persona}</b> (cluster {cluster_id})<br>
                    Suggested action: <b style="color:{COLORS['text']};">
                    {"Immediate retention outreach — offer + service recovery" if risk_label=="High" else
                     "Proactive engagement — personalized offer or check-in" if risk_label=="Medium" else
                     "Maintain experience — nurture loyalty program"}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE 5: FEATURE IMPORTANCE
# --------------------------------------------------------------------------
def page_feature_importance():
    section_header("Feature Importance", "Which behavioral signals drive the Random Forest's churn predictions", "🔥")

    fi = bundle["feature_importance"].copy()
    fi["Importance_%"] = fi["Importance"] * 100

    top_n = st.slider("Number of top features to display", 5, len(fi), 12)
    top_fi = fi.head(top_n).sort_values("Importance")

    fig = go.Figure(go.Bar(
        x=top_fi["Importance_%"], y=top_fi["Feature"], orientation="h",
        marker=dict(color=top_fi["Importance_%"], colorscale=[[0, COLORS["primary"]], [1, COLORS["teal"]]]),
        text=top_fi["Importance_%"].round(1).astype(str) + "%", textposition="outside",
    ))
    fig.update_layout(title=f"Top {top_n} Features Driving Churn Prediction", xaxis_title="Relative Importance (%)")
    st.plotly_chart(style_fig(fig, max(380, 26 * top_n)), use_container_width=True)

    section_header("Cumulative Importance", "How many features are needed to explain most of the model's decisions", "📶")
    fi_sorted = fi.sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_sorted["Cumulative"] = fi_sorted["Importance"].cumsum() * 100
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(fi_sorted) + 1)), y=fi_sorted["Cumulative"],
        mode="lines+markers", line=dict(color=COLORS["teal"], width=3),
    ))
    fig.add_hline(y=80, line_dash="dash", line_color=COLORS["amber"],
                  annotation_text="80% threshold", annotation_font_color=COLORS["amber"])
    fig.update_layout(title="Cumulative Feature Importance", xaxis_title="Number of Features", yaxis_title="Cumulative Importance (%)")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    top3 = fi_sorted.head(3)["Feature"].tolist()
    n_for_80 = int((fi_sorted["Cumulative"] >= 80).idxmax() + 1)
    st.markdown(f"""
    <div class="insight-card">
    The top 3 churn drivers are <b>{top3[0]}</b>, <b>{top3[1]}</b>, and <b>{top3[2]}</b>.
    Just <b>{n_for_80}</b> of {len(fi_sorted)} features explain 80% of the model's predictive power —
    a focused set of behavioral and financial signals to monitor for early churn warning.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Full Feature Importance Table"):
        st.dataframe(fi[["Feature", "Importance_%"]].round(3), use_container_width=True, height=400)


# --------------------------------------------------------------------------
# PAGE 6: INSIGHTS & RECOMMENDATIONS
# --------------------------------------------------------------------------
def page_insights():
    section_header("Customer Insights & Recommendations", "Data-driven findings and a segment-by-segment retention playbook", "💡")

    profile = bundle["cluster_profile"].sort_values("Churn_Rate_%", ascending=False)
    fi = bundle["feature_importance"]
    top_feature = fi.iloc[0]["Feature"]
    riskiest = profile.iloc[0]
    healthiest = profile.sort_values("Churn_Rate_%").iloc[0]
    overall_churn = df["Churned"].mean() * 100
    at_risk_value = (riskiest["Size"] * riskiest["Lifetime_Value"])
    high_abandon_corr = df[["Cart_Abandonment_Rate", "Churned"]].corr().iloc[0, 1]

    insights = [
        f"Overall churn stands at <b>{overall_churn:.1f}%</b> across {len(df):,} customers, "
        f"with the <b>{riskiest['Persona']}</b> segment carrying the highest risk at "
        f"<b>{riskiest['Churn_Rate_%']:.1f}%</b> churn.",

        f"<b>{top_feature.replace('_',' ')}</b> is the single strongest predictor of churn according to the "
        f"Random Forest model, ahead of every other behavioral or demographic signal.",

        f"Cart abandonment rate correlates with churn at <b>{high_abandon_corr:.2f}</b> — "
        f"customers who repeatedly abandon carts are measurably more likely to leave.",

        f"The <b>{riskiest['Persona']}</b> segment alone represents roughly "
        f"<b>${at_risk_value:,.0f}</b> in cumulative lifetime value that is currently at elevated churn risk.",

        f"The <b>{healthiest['Persona']}</b> segment shows the lowest churn at "
        f"<b>{healthiest['Churn_Rate_%']:.1f}%</b> and the highest average lifetime value of the four personas — "
        f"a benchmark for what healthy engagement looks like.",
    ]
    for txt in insights:
        st.markdown(f'<div class="insight-card">🔎 {txt}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_header("Segment Retention Playbook", "Recommended action per persona", "🧭")

    playbook = {
        "Champions": [
            "Protect the relationship with a VIP/loyalty tier and early access to new products.",
            "Use as advocates — referral incentives and review requests.",
            "Monitor for any dip in login frequency as an early warning signal.",
        ],
        "Loyal Customers": [
            "Upsell and cross-sell based on purchase history to grow lifetime value.",
            "Reward consistency with tiered loyalty points to reinforce the habit.",
            "Introduce subscription or auto-replenishment options where relevant.",
        ],
        "At-Risk Customers": [
            "Trigger proactive win-back campaigns with personalized discounts.",
            "Reduce friction at checkout — cart abandonment is a key churn driver here.",
            "Route to customer service proactively before complaints escalate.",
        ],
        "Dormant / Low-Engagement": [
            "Run a re-engagement email/push sequence with a strong incentive to return.",
            "Survey to understand disengagement before investing further marketing spend.",
            "Consider suppressing low-value dormant accounts from paid acquisition lookalikes.",
        ],
    }

    cols = st.columns(2)
    for i, (_, row) in enumerate(profile.sort_values("Cluster").iterrows()):
        persona = row["Persona"]
        accent = PERSONA_COLORS.get(persona, COLORS["primary"])
        actions = playbook.get(persona, [])
        actions_html = "".join([f"<li>{a}</li>" for a in actions])
        with cols[i % 2]:
            st.markdown(f"""
            <div class="reco-card" style="border-left:4px solid {accent};">
                <div class="persona-title">{persona}</div>
                <ul class="persona-desc" style="padding-left:18px; margin-top:8px;">{actions_html}</ul>
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE 7: REPORTS & EXPORT
# --------------------------------------------------------------------------
def build_pdf_report():
    from fpdf import FPDF

    m = bundle["metrics"]
    profile = bundle["cluster_profile"].sort_values("Cluster")
    fi = bundle["feature_importance"].head(10)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(17, 17, 40)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "ChurnIQ - Customer Behavior Analytics Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 20)
    pdf.cell(0, 6, f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", ln=True)

    pdf.set_text_color(20, 20, 20)
    pdf.ln(18)

    def h(title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 90)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "", 10.5)

    h("Executive Summary")
    pdf.multi_cell(0, 6,
        f"Total customers analyzed: {len(df):,}\n"
        f"Overall churn rate: {df['Churned'].mean()*100:.1f}%\n"
        f"Average lifetime value: ${df['Lifetime_Value'].mean():,.0f}\n"
        f"Customer segments identified: {bundle['n_clusters']} (K-Means, silhouette={bundle['silhouette_score']:.3f})\n"
    )
    pdf.ln(2)

    h("Predictive Model Performance (Random Forest)")
    pdf.multi_cell(0, 6,
        f"Accuracy: {m['accuracy']*100:.1f}%   Precision: {m['precision']*100:.1f}%   "
        f"Recall: {m['recall']*100:.1f}%   F1: {m['f1_score']*100:.1f}%   ROC-AUC: {m['roc_auc']:.3f}\n"
    )
    pdf.ln(2)

    h("Customer Segments")
    for _, row in profile.iterrows():
        pdf.multi_cell(0, 6,
            f"- {row['Persona']}: {int(row['Size']):,} customers, "
            f"{row['Churn_Rate_%']:.1f}% churn rate, avg LTV ${row['Lifetime_Value']:,.0f}"
        )
    pdf.ln(2)

    h("Top Churn-Driving Features")
    for _, row in fi.iterrows():
        pdf.multi_cell(0, 6, f"- {row['Feature']}: {row['Importance']*100:.1f}% relative importance")
    pdf.ln(2)

    h("Key Recommendations")
    pdf.multi_cell(0, 6,
        "1. Prioritize retention outreach for At-Risk and Dormant segments.\n"
        "2. Reduce cart abandonment friction at checkout - a top churn driver.\n"
        "3. Reward and protect the Champions segment through loyalty perks.\n"
        "4. Use the live prediction tool to proactively score new customers monthly.\n"
    )

    return bytes(pdf.output(dest="S"))


def page_reports():
    section_header("Reports & Export", "Download processed data, segment summaries, and an executive PDF report", "📥")

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Total Records", f"{len(df):,}", "🗂️")
    with c2: kpi_card("Model Accuracy", f"{bundle['metrics']['accuracy']*100:.1f}%", "🎯")
    with c3: kpi_card("Report Generated", datetime.now().strftime("%d %b %Y"), "📅")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><b>📄 Processed Dataset</b><br><span style="font-size:13px;color:#93A0C2;">Full customer table incl. cluster, persona, and churn probability.</span></div>', unsafe_allow_html=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                            file_name="processed_customer_data.csv", mime="text/csv", use_container_width=True)

    with col2:
        st.markdown('<div class="card"><b>🧩 Segment Summary</b><br><span style="font-size:13px;color:#93A0C2;">Cluster profiles with persona, size, churn rate, and value.</span></div>', unsafe_allow_html=True)
        seg_csv = bundle["cluster_profile"].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", seg_csv, file_name="segment_summary.csv",
                            mime="text/csv", use_container_width=True)

    with col3:
        st.markdown('<div class="card"><b>📑 Executive PDF Report</b><br><span style="font-size:13px;color:#93A0C2;">One-page summary of KPIs, model results, and recommendations.</span></div>', unsafe_allow_html=True)
        try:
            pdf_bytes = build_pdf_report()
            st.download_button("Download PDF", pdf_bytes, file_name="churniq_executive_report.pdf",
                                mime="application/pdf", use_container_width=True)
        except ImportError:
            st.warning("Install `fpdf2` (see requirements.txt) to enable PDF export.")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)
    section_header("Report Preview", "Quick summary of everything the PDF report contains", "👁️")
    profile = bundle["cluster_profile"].sort_values("Cluster")
    st.dataframe(
        profile[["Persona", "Size", "Churn_Rate_%", "Lifetime_Value", "Average_Order_Value"]]
        .rename(columns={"Churn_Rate_%": "Churn Rate (%)", "Lifetime_Value": "Avg LTV ($)", "Average_Order_Value": "Avg Order Value ($)"})
        .round(1),
        use_container_width=True,
    )


# --------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------
PAGE_FUNCS = {
    "Overview": page_overview,
    "Exploratory Data Analysis": page_eda,
    "Customer Segmentation": page_segmentation,
    "Predictive Modelling": page_predictive,
    "Feature Importance": page_feature_importance,
    "Insights & Recommendations": page_insights,
    "Reports & Export": page_reports,
}

PAGE_FUNCS[selected]()
