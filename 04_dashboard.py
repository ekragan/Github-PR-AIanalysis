"""
Script 4: GitHub PR Analysis Dashboard
========================================
A Streamlit web dashboard that visualises:
  - PR stats per repo (failures, merge rate, review activity)
  - Author breakdown
  - The full AI analysis report
  - Key health indicators

Setup:
    pip install streamlit pandas plotly

Usage:
    streamlit run 04_dashboard.py
    (opens automatically in your browser at http://localhost:8501)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PR_DATA_FILE    = "pr_data.csv"
REPORT_FILE     = "analysis_report.txt"
# ─────────────────────────────────────────────────────────────


# ── Page config — must be the first Streamlit call ──────────
st.set_page_config(
    page_title  = "GitHub PR Intelligence",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "collapsed"
)

# ── Custom CSS — dark engineering aesthetic ──────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

  /* Base */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d14;
    color: #e8e8f0;
  }
  .stApp { background-color: #0d0d14; }

  /* Header */
  .dash-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
  }
  .dash-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #ffffff;
    margin: 0 0 4px;
    letter-spacing: -0.5px;
  }
  .dash-header p {
    color: #7a7aaa;
    font-size: 0.85rem;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Metric cards */
  .metric-card {
    background: #13131f;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #7eb8f7;
    line-height: 1;
    margin-bottom: 6px;
  }
  .metric-value.green  { color: #6ee7b7; }
  .metric-value.red    { color: #f87171; }
  .metric-value.yellow { color: #fbbf24; }
  .metric-label {
    font-size: 0.75rem;
    color: #6666aa;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Section headers */
  .section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4444aa;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #2a2a4a;
  }

  /* Report box */
  .report-box {
    background: #0a0a12;
    border: 1px solid #2a2a4a;
    border-left: 3px solid #7eb8f7;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    color: #c0c0d8;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  /* Hide Streamlit branding */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }

  /* Chart backgrounds */
  .js-plotly-plot { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data  # cache_data = Streamlit caches this so it doesn't reload on every interaction
def load_data():
    if not os.path.exists(PR_DATA_FILE):
        return None
    df = pd.read_csv(PR_DATA_FILE)
    return df

@st.cache_data
def load_report():
    if not os.path.exists(REPORT_FILE):
        return None
    with open(REPORT_FILE, "r") as f:
        return f.read()

df     = load_data()
report = load_report()

# ─────────────────────────────────────────────────────────────
# PLOTLY CHART THEME — dark background to match dashboard
# ─────────────────────────────────────────────────────────────
CHART_THEME = dict(
    plot_bgcolor  = "#13131f",
    paper_bgcolor = "#13131f",
    font_color    = "#a0a0c0",
    font_family   = "JetBrains Mono",
)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>🔬 GitHub PR Intelligence</h1>
  <p>AI-powered analysis of your pull request health & team patterns</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# GUARD: show a message if data files are missing
# ─────────────────────────────────────────────────────────────
if df is None:
    st.error(f"**{PR_DATA_FILE} not found.** Run `01_collect_github_data.py` first.")
    st.stop()  # stop rendering the rest of the dashboard


# ─────────────────────────────────────────────────────────────
# SECTION 1 — HEADLINE METRICS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">01 — Overview</div>', unsafe_allow_html=True)

total_prs      = len(df)
total_repos    = df["repo"].nunique()
total_merged   = int(df["is_merged"].sum())
merge_rate     = round(total_merged / total_prs * 100, 1) if total_prs > 0 else 0
ci_failures    = int((df["ci_status"] == "failed").sum())
failure_rate   = round(ci_failures / total_prs * 100, 1) if total_prs > 0 else 0
avg_hours_open = round(df["hours_open"].mean(), 1)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_prs}</div><div class="metric-label">Total PRs</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_repos}</div><div class="metric-label">Repos Scanned</div></div>', unsafe_allow_html=True)
with col3:
    color = "green" if merge_rate >= 80 else "yellow" if merge_rate >= 50 else "red"
    st.markdown(f'<div class="metric-card"><div class="metric-value {color}">{merge_rate}%</div><div class="metric-label">Merge Rate</div></div>', unsafe_allow_html=True)
with col4:
    color = "green" if ci_failures == 0 else "red" if failure_rate > 30 else "yellow"
    st.markdown(f'<div class="metric-card"><div class="metric-value {color}">{ci_failures}</div><div class="metric-label">CI Failures</div></div>', unsafe_allow_html=True)
with col5:
    color = "green" if avg_hours_open < 24 else "yellow" if avg_hours_open < 72 else "red"
    st.markdown(f'<div class="metric-card"><div class="metric-value {color}">{avg_hours_open}h</div><div class="metric-label">Avg Hours Open</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECTION 2 — CHARTS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">02 — Repo Breakdown</div>', unsafe_allow_html=True)

# Build per-repo stats
repo_stats = df.groupby("repo").agg(
    total_prs    = ("pr_number", "count"),
    merged       = ("is_merged", "sum"),
    ci_failures  = ("ci_status", lambda x: (x == "failed").sum()),
    avg_review_comments = ("review_comments", "mean"),
    avg_hours_open = ("hours_open", "mean"),
).reset_index()
repo_stats["failure_rate_pct"] = (repo_stats["ci_failures"] / repo_stats["total_prs"] * 100).round(1)
repo_stats["merge_rate_pct"]   = (repo_stats["merged"] / repo_stats["total_prs"] * 100).round(1)

col_left, col_right = st.columns(2)

with col_left:
    # Bar chart: PRs per repo, split by merged vs not merged
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="Merged",
        x=repo_stats["repo"],
        y=repo_stats["merged"],
        marker_color="#6ee7b7",
        marker_line_width=0,
    ))
    fig1.add_trace(go.Bar(
        name="Not Merged",
        x=repo_stats["repo"],
        y=repo_stats["total_prs"] - repo_stats["merged"],
        marker_color="#f87171",
        marker_line_width=0,
    ))
    fig1.update_layout(
        **CHART_THEME,
        barmode="stack",
        title=dict(text="PRs by Repo — Merged vs Not Merged", font_color="#e8e8f0", font_size=13),
        legend=dict(bgcolor="#13131f", font_color="#a0a0c0"),
        xaxis=dict(gridcolor="#1e1e32"),
        yaxis=dict(gridcolor="#1e1e32"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=300,
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    # Bar chart: CI failure rate per repo
    fig2 = px.bar(
        repo_stats,
        x="repo",
        y="failure_rate_pct",
        color="failure_rate_pct",
        color_continuous_scale=["#6ee7b7", "#fbbf24", "#f87171"],
        labels={"failure_rate_pct": "Failure Rate %", "repo": "Repo"},
        title="CI Failure Rate % per Repo",
    )
    fig2.update_layout(
        **CHART_THEME,
        title=dict(font_color="#e8e8f0", font_size=13),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#1e1e32"),
        yaxis=dict(gridcolor="#1e1e32"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=300,
    )
    fig2.update_traces(marker_line_width=0)
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# SECTION 3 — AUTHOR BREAKDOWN
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">03 — Author Patterns</div>', unsafe_allow_html=True)

author_stats = df.groupby("author").agg(
    prs_opened   = ("pr_number", "count"),
    ci_failures  = ("ci_status", lambda x: (x == "failed").sum()),
    avg_review_comments = ("review_comments", "mean"),
).reset_index().sort_values("prs_opened", ascending=False).head(10)

col_a, col_b = st.columns([2, 1])

with col_a:
    fig3 = px.bar(
        author_stats,
        x="author",
        y="prs_opened",
        color="ci_failures",
        color_continuous_scale=["#6ee7b7", "#fbbf24", "#f87171"],
        labels={"prs_opened": "PRs Opened", "author": "Author", "ci_failures": "CI Failures"},
        title="PRs per Author (colour = CI failures)",
    )
    fig3.update_layout(
        **CHART_THEME,
        title=dict(font_color="#e8e8f0", font_size=13),
        xaxis=dict(gridcolor="#1e1e32"),
        yaxis=dict(gridcolor="#1e1e32"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=300,
    )
    fig3.update_traces(marker_line_width=0)
    st.plotly_chart(fig3, use_container_width=True)

with col_b:
    # Show the author table cleanly
    st.dataframe(
        author_stats.rename(columns={
            "author": "Author",
            "prs_opened": "PRs",
            "ci_failures": "CI Fails",
            "avg_review_comments": "Avg Reviews",
        }),
        use_container_width=True,
        hide_index=True,
        height=300,
    )


# ─────────────────────────────────────────────────────────────
# SECTION 4 — PR TIMELINE (if created_at exists)
# ─────────────────────────────────────────────────────────────
if "created_at" in df.columns:
    st.markdown('<div class="section-header">04 — PR Activity Over Time</div>', unsafe_allow_html=True)

    df["created_at"] = pd.to_datetime(df["created_at"])
    timeline = df.groupby(df["created_at"].dt.to_period("M")).size().reset_index()
    timeline.columns = ["month", "pr_count"]
    timeline["month"] = timeline["month"].astype(str)

    fig4 = px.line(
        timeline,
        x="month",
        y="pr_count",
        markers=True,
        labels={"month": "Month", "pr_count": "PRs Opened"},
        title="PR Volume Over Time",
    )
    fig4.update_layout(
        **CHART_THEME,
        title=dict(font_color="#e8e8f0", font_size=13),
        xaxis=dict(gridcolor="#1e1e32"),
        yaxis=dict(gridcolor="#1e1e32"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=260,
    )
    fig4.update_traces(line_color="#7eb8f7", marker_color="#7eb8f7")
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# SECTION 5 — AI ANALYSIS REPORT
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">05 — AI Analysis Report</div>', unsafe_allow_html=True)

if report:
    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
else:
    st.warning(
        f"**{REPORT_FILE} not found.**  \n"
        "Run `02_analyse_with_ai.py` and paste Claude's response into `analysis_report.txt`."
    )


# ─────────────────────────────────────────────────────────────
# SECTION 6 — RAW DATA TABLE (expandable)
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">06 — Raw PR Data</div>', unsafe_allow_html=True)

with st.expander("View all PR records"):
    st.dataframe(df, use_container_width=True, hide_index=True)