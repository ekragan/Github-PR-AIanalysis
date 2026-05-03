"""
Script 2: AI Analysis of PR Data
==================================
This script is the "brain" of our POC. It does 3 things:
  1. Reads the pr_data.csv that Script 1 collected
  2. Summarises it into meaningful stats (using pandas)
  3. Sends those stats to an AI (Claude or OpenAI) and gets back a human-readable report

KEY CONCEPT — Why summarise before sending to AI?
    LLMs have a "context window" — a limit on how much text they can read at once.
    Sending 500 raw CSV rows would be wasteful and expensive.
    Instead, we crunch the data into a compact summary first, then send that.
    This is a core GenAI pattern called "data-to-prompt compression".

Setup:
    pip install anthropic pandas        # if using Claude
    pip install openai pandas           # if using OpenAI

Usage:
    1. Set your API key and preferred LLM in the CONFIG section below
    2. Make sure pr_data.csv is in the same folder (run Script 1 first)
    3. Run: python 02_analyse_with_ai.py
"""

import pandas as pd   # pandas = the go-to library for working with tabular data (like spreadsheets)
import json           # json = used to format Python dicts into clean readable text for the prompt


# ─────────────────────────────────────────────────────────────
# CONFIG — the only section you need to edit before running
# ─────────────────────────────────────────────────────────────
LLM_PROVIDER  = "claude"              # Which AI to use: "claude" or "openai"
API_KEY       = "your_api_key_here"   # Only needed if FREE_MODE = False
INPUT_FILE    = "pr_data.csv"         # The CSV produced by Script 1
OUTPUT_FILE   = "analysis_report.txt" # Where to save the AI report
PROMPT_FILE   = "prompt_for_claude.txt" # Where to save the prompt in FREE_MODE

# ── FREE MODE ──────────────────────────────────────────────────
# Have Claude Pro at claude.ai but no API key? Set FREE_MODE = True
# The script will:
#   Step 1 → crunch your PR data into stats (automatic)
#   Step 2 → save the full prompt to "prompt_for_claude.txt" (automatic)
#   Step 3 → YOU open claude.ai, paste the prompt, copy the response
#   Step 4 → paste Claude's response into "analysis_report.txt" (manual)
#   Step 5 → run Script 3 to email the report (automatic)
#
# Once you get an API key, just set FREE_MODE = False and it all runs automatically.
FREE_MODE = True   # ← change to False once you have an API key
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load the CSV and compute statistics
# ─────────────────────────────────────────────────────────────
def load_and_summarise(filepath):
    """
    Reads the raw PR data and aggregates it into 3 summaries:
      - repo_stats   : one row per repo with failure counts, review activity etc.
      - author_stats : one row per author with how many PRs they opened
      - overall      : a single dict of headline numbers for the whole dataset

    WHY THIS STEP?
    Raw CSV rows look like: "repo=myapp, pr=42, author=alice, ci_status=failed ..."
    That's not very useful to an LLM. Aggregating it gives us:
    "myapp had 15 PRs, 8 CI failures (53% failure rate), avg 3.2 review comments"
    — which is exactly what an engineering manager would want to see.
    """

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} PRs across {df['repo'].nunique()} repos\n")

    # ── Per-repo stats ──────────────────────────────────────────
    # groupby("repo") = group all rows that share the same repo name
    # .agg(...)       = for each group, compute these aggregations
    repo_stats = df.groupby("repo").agg(
        total_prs           = ("pr_number", "count"),        # count of all PRs in this repo
        merged              = ("is_merged", "sum"),           # how many were successfully merged
        ci_failures         = ("ci_status", lambda x: (x == "failed").sum()),  # count where CI failed
        avg_review_comments = ("review_comments", "mean"),   # average review comments per PR
        avg_hours_open      = ("hours_open", "mean"),        # average time a PR stayed open
        total_comments      = ("total_comments", "sum"),     # total discussion volume
    ).reset_index()  # reset_index turns the grouped index back into a regular column

    # Derived metric: what % of PRs in this repo had CI failures?
    repo_stats["failure_rate_pct"] = (
        repo_stats["ci_failures"] / repo_stats["total_prs"] * 100
    ).round(1)

    repo_stats["avg_review_comments"] = repo_stats["avg_review_comments"].round(1)
    repo_stats["avg_hours_open"]      = repo_stats["avg_hours_open"].round(1)

    # Sort so the worst-performing repos appear first — helps the LLM focus on them
    repo_stats = repo_stats.sort_values("ci_failures", ascending=False)

    # ── Per-author stats ────────────────────────────────────────
    # Helps identify if one person is responsible for many failures
    # or if a reviewer is a bottleneck (everyone waits for them)
    author_stats = df.groupby("author").agg(
        prs_opened          = ("pr_number", "count"),
        ci_failures         = ("ci_status", lambda x: (x == "failed").sum()),
        avg_review_comments = ("review_comments", "mean"),
    ).reset_index()
    author_stats["avg_review_comments"] = author_stats["avg_review_comments"].round(1)
    author_stats = author_stats.sort_values("prs_opened", ascending=False).head(10)  # top 10 only

    # ── Overall headline stats ──────────────────────────────────
    # A quick snapshot of the entire dataset — goes at the top of the AI prompt
    overall = {
        "total_prs"          : len(df),
        "total_repos"        : df["repo"].nunique(),
        "total_merged"       : int(df["is_merged"].sum()),
        "total_ci_failures"  : int((df["ci_status"] == "failed").sum()),
        "avg_hours_open"     : round(df["hours_open"].mean(), 1),
        "avg_review_comments": round(df["review_comments"].mean(), 1),
        "most_active_author" : df["author"].value_counts().idxmax(),  # author with most PRs
    }

    return df, repo_stats, author_stats, overall


# ─────────────────────────────────────────────────────────────
# STEP 2 — Build the prompt
# ─────────────────────────────────────────────────────────────
def build_prompt(repo_stats, author_stats, overall):
    """
    Converts our pandas dataframes into a text prompt for the LLM.

    KEY CONCEPT — Prompt Engineering:
    The way you write a prompt dramatically affects the quality of the AI response.
    Good prompts have 4 ingredients:
      1. ROLE        — tell the AI who it is ("You are a senior engineering manager...")
      2. CONTEXT     — give it the actual data to work with
      3. TASK        — clearly numbered list of what to analyse
      4. FORMAT      — specify how you want the output structured

    Think of this function as writing the "brief" you'd hand to a consultant.
    The better the brief, the better the report you get back.
    """

    # .to_string(index=False) converts a DataFrame to a neat plain-text table
    # LLMs read plain text tables well — better than raw CSV
    repo_table   = repo_stats.to_string(index=False)
    author_table = author_stats.to_string(index=False)

    # json.dumps with indent=2 formats a Python dict as nicely indented JSON text
    # LLMs understand JSON very well — it's a clean, structured format
    prompt = f"""
You are a senior engineering manager reviewing GitHub Pull Request data for a software team.
Your job is to analyse the statistics below and give the team practical, data-driven insights.

=== OVERALL SUMMARY ===
{json.dumps(overall, indent=2)}

=== PER-REPO STATS (sorted by CI failures, highest first) ===
{repo_table}

=== TOP AUTHORS BY PR VOLUME ===
{author_table}

=== YOUR ANALYSIS TASK ===
Please provide a structured report with the following sections:

1. TOP FAILING REPOS
   - Which repos have the highest CI failure rate?
   - What are the likely root causes? (e.g. missing tests, flaky pipelines, large PRs)

2. MOST REVIEWED REPOS
   - Which repos have the highest review activity?
   - Is heavy review a sign of complexity, knowledge sharing, or a bottleneck?

3. AUTHOR PATTERNS
   - Are any authors creating PRs with unusually high failure or review rates?
   - Any signs of authors who might need support or mentoring?

4. PR HEALTH INDICATORS
   - Comment on the average time PRs stay open
   - Are PRs sitting too long? What does that suggest about the process?

5. RECOMMENDED ACTIONS
   - Give exactly 3 specific, actionable recommendations the team can act on this sprint

Use the actual numbers from the data in your analysis. Keep the tone constructive.
"""
    return prompt.strip()


# ─────────────────────────────────────────────────────────────
# STEP 3a — Call Claude (Anthropic)
# ─────────────────────────────────────────────────────────────
def call_claude(api_key, prompt):
    """
    Sends the prompt to Claude via Anthropic's API and returns the response text
    AND token usage stats so we can estimate cost.

    HOW AN LLM API CALL WORKS:
      1. We create a client using our API key (like logging into a service)
      2. We send a "messages" list — this is the conversation format all LLMs use
           role: "user"      = our message (the prompt we built)
           role: "assistant" = Claude's previous replies (for multi-turn chats)
      3. max_tokens limits how long the response can be (1 token ≈ 0.75 words)
      4. The response comes back as a list of content blocks — we grab [0].text

    COST AWARENESS:
      The API response includes a usage object with exact token counts:
        - input_tokens  : tokens in the prompt we sent
        - output_tokens : tokens in the response we received
      We use these to calculate the exact cost of each run.

    CLAUDE SONNET PRICING (as of 2025):
      Input  : $3.00 per 1 million tokens  → $0.000003 per token
      Output : $15.00 per 1 million tokens → $0.000015 per token
    """
    import anthropic  # imported here so the script still works if only openai is installed

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model      = "claude-opus-4-5",  # the specific Claude model to use
        max_tokens = 1500,               # cap the response so it doesn't go on forever
        messages   = [
            {"role": "user", "content": prompt}  # our prompt goes here as the user turn
        ]
    )

    # message.usage contains the exact token counts for this API call
    # We return these alongside the text so main() can print the cost breakdown
    usage = message.usage

    return message.content[0].text, usage.input_tokens, usage.output_tokens


# ─────────────────────────────────────────────────────────────
# STEP 3b — Call OpenAI (alternative to Claude)
# ─────────────────────────────────────────────────────────────
def call_openai(api_key, prompt):
    """
    Same idea as call_claude() but using OpenAI's API instead.

    The API structure is slightly different:
      - OpenAI uses:  client.chat.completions.create()
      - Response via: response.choices[0].message.content

    But both use the same "messages" list format with role/content pairs.
    This standard is called the "Chat Completions" format and most LLM providers follow it.
    Once you know one, switching to another is straightforward.
    """
    from openai import OpenAI  # imported here so script works if only anthropic is installed

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model      = "gpt-4o",   # the specific OpenAI model to use
        max_tokens = 1500,
        messages   = [
            {"role": "user", "content": prompt}
        ]
    )

    # OpenAI also returns token usage — same pattern as Claude
    usage = response.usage
    return response.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens


# ─────────────────────────────────────────────────────────────
# MAIN — ties all the steps together in order
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  GitHub PR AI Analyser")
    print(f"  Mode: {'FREE (Claude Pro)' if FREE_MODE else 'API KEY'}")
    print("=" * 55)

    # ── Step 1: Load CSV and compute stats ──────────────────────
    print(f"\n  Step 1: Reading {INPUT_FILE}...")
    try:
        df, repo_stats, author_stats, overall = load_and_summarise(INPUT_FILE)
    except FileNotFoundError:
        print(f"\n  ERROR: {INPUT_FILE} not found.")
        print("  Please run 01_collect_github_data.py first to generate it.")
        return

    # ── Step 2: Build the prompt ─────────────────────────────────
    print("  Step 2: Building prompt from stats...")
    prompt = build_prompt(repo_stats, author_stats, overall)
    print(f"  Prompt ready — {len(prompt)} characters, ~{len(prompt)//4} tokens")

    # ════════════════════════════════════════════════════════════
    # FREE MODE — no API key needed, uses your Claude Pro account
    # ════════════════════════════════════════════════════════════
    if FREE_MODE:
        # Save the prompt to a text file so you can copy-paste it into claude.ai
        with open(PROMPT_FILE, "w") as f:
            f.write(prompt)

        print("\n" + "=" * 55)
        print("  FREE MODE — 3 manual steps for you:")
        print("=" * 55)
        print(f"\n  1. Open this file: {PROMPT_FILE}")
        print("     (it contains your full analysis prompt)")
        print("\n  2. Go to claude.ai → paste the entire contents → hit Enter")
        print("     (Claude Pro will analyse your PR data for free)")
        print(f"\n  3. Copy Claude's full response and paste it into: {OUTPUT_FILE}")
        print("     (create the file if it doesn't exist)")
        print("\n  Then run:  python 03_send_email.py  to email the report!")
        print("=" * 55)
        print("\n  TIP: Once you get an API key from console.anthropic.com,")
        print("  set FREE_MODE = False in CONFIG and everything runs automatically.")
        return

    # ════════════════════════════════════════════════════════════
    # API MODE — fully automatic, requires API key in CONFIG
    # ════════════════════════════════════════════════════════════

    # Guard clause: remind the user to fill in their API key
    if API_KEY == "your_api_key_here":
        print("\n  ERROR: FREE_MODE is False but no API_KEY is set.")
        print("  Either set FREE_MODE = True  (use Claude Pro manually)")
        print("  Or set API_KEY to your key from console.anthropic.com")
        return

    # ── Step 3: Send to LLM ──────────────────────────────────────
    print(f"\n  Step 3: Sending to {LLM_PROVIDER.upper()} — please wait...")
    try:
        if LLM_PROVIDER == "claude":
            report, input_tokens, output_tokens = call_claude(API_KEY, prompt)
        else:
            report, input_tokens, output_tokens = call_openai(API_KEY, prompt)
    except Exception as e:
        print(f"\n  ERROR while calling {LLM_PROVIDER}: {e}")
        print("  Check your API key and internet connection.")
        return

    # ── Token cost breakdown ─────────────────────────────────────
    # Claude Opus pricing: $15 input / $75 output per 1M tokens (as of 2025)
    # OpenAI GPT-4o pricing: $5 input / $15 output per 1M tokens (as of 2025)
    if LLM_PROVIDER == "claude":
        input_cost  = (input_tokens  / 1_000_000) * 15.00
        output_cost = (output_tokens / 1_000_000) * 75.00
    else:
        input_cost  = (input_tokens  / 1_000_000) * 5.00
        output_cost = (output_tokens / 1_000_000) * 15.00

    total_cost = input_cost + output_cost

    print(f"\n  Token usage:")
    print(f"    Input tokens   : {input_tokens:,}   (prompt we sent)")
    print(f"    Output tokens  : {output_tokens:,}  (response we got)")
    print(f"    Total tokens   : {input_tokens + output_tokens:,}")
    print(f"\n  Estimated cost this run:")
    print(f"    Input cost     : ${input_cost:.6f}")
    print(f"    Output cost    : ${output_cost:.6f}")
    print(f"    TOTAL          : ${total_cost:.6f}  (~₹{total_cost * 83:.4f} INR)")

    # ── Step 4: Print and save the report ────────────────────────
    print("\n" + "=" * 55)
    print("  AI ANALYSIS REPORT")
    print("=" * 55)
    print(report)

    with open(OUTPUT_FILE, "w") as f:
        f.write("GitHub PR AI Analysis Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)

    print("\n" + "=" * 55)
    print(f"  Report saved to: {OUTPUT_FILE}")
    print(f"  Next: run 03_send_email.py to email the report!")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────
# Python best practice: only run main() if this file is executed
# directly (e.g. python 02_analyse_with_ai.py), NOT when it's
# imported as a module by another script.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()