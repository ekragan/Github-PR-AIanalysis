import pandas as pd
import json
import os
from openai import OpenAI

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
INPUT_FILE = "pr_data.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

print("\nLoaded PR data:", len(df), "rows")

# ──────────────────────────────────────────────
# BUILD SUMMARY (VERY IMPORTANT STEP)
# ──────────────────────────────────────────────

summary = {}

summary["total_prs"] = len(df)
summary["merged_prs"] = int(df["is_merged"].sum())
summary["ci_failures"] = int((df["ci_status"] == "failed").sum())

# Repo-wise stats
repo_group = df.groupby("repo")

summary["repo_stats"] = {}

for repo, group in repo_group:
    summary["repo_stats"][repo] = {
        "total_prs": int(len(group)),
        "ci_failures": int((group["ci_status"] == "failed").sum()),
        "avg_review_comments": round(group["review_comments"].mean(), 2),
        "avg_time_open_hours": round(group["hours_open"].mean(), 2)
    }

# Top repos by friction (reviews + time)
df["review_friction"] = df["review_comments"] + df["total_comments"]

top_friction = (
    df.groupby("repo")["review_friction"]
    .mean()
    .sort_values(ascending=False)
    .head(3)
    .to_dict()
)

summary["high_friction_repos"] = top_friction

# Slow PRs
slow_prs = df[df["hours_open"] > 72]
summary["slow_prs_count"] = int(len(slow_prs))

# ──────────────────────────────────────────────
# BUILD PROMPT
# ──────────────────────────────────────────────

prompt = f"""
You are analyzing GitHub PR and CI data for engineering productivity.

Here is structured data:
{json.dumps(summary, indent=2)}

Analyse and provide:

1. Top 3 recurring failure patterns
2. Repositories with high review friction (many comments + long open time)
3. Potential bottlenecks (repos or contributors)
4. 2-3 actionable recommendations to improve CI reliability and PR flow

Be concise. Use bullet points.
"""

# ──────────────────────────────────────────────
# CALL LLM
# ──────────────────────────────────────────────

print("\nCalling AI model...\n")

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are an expert in DevOps and engineering productivity."},
        {"role": "user", "content": prompt}
    ]
)

analysis = response.choices[0].message.content

# ──────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────

print("\n" + "="*60)
print("AI ANALYSIS")
print("="*60)
print(analysis)

# Save output
with open("ai_analysis.txt", "w") as f:
    f.write(analysis)

print("\nSaved to ai_analysis.txt")