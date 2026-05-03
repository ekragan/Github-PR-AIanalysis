"""
Script 1: GitHub Data Collector
================================
Fetches PR data from your GitHub repos and saves it to a CSV file.

Setup:
    pip install PyGithub pandas

Usage:
    1. Replace GITHUB_TOKEN with your token
    2. Replace GITHUB_USERNAME with your GitHub username (or org name)
    3. Run: python 01_collect_github_data.py
"""

import os
import pandas as pd
from github import Github
from datetime import datetime, timezone

# ──────────────────────────────────────────────
# CONFIG — edit these before running
# ──────────────────────────────────────────────
GITHUB_TOKEN    = "<add yours>"   # GitHub PAT (Settings > Developer settings > Tokens)
GITHUB_USERNAME = "<add yours>"    # Your GitHub username or org name
MAX_REPOS       = 10                      # How many repos to scan (increase as needed)
MAX_PRS_PER_REPO = 50                     # PRs to fetch per repo
PR_STATE        = "all"                   # "open", "closed", or "all"
OUTPUT_FILE     = "pr_data.csv"
# ──────────────────────────────────────────────


def get_check_run_status(pr):
    """Get the overall CI status for a PR's last commit."""
    try:
        commit = pr.get_commits().reversed[0]
        check_runs = list(commit.get_check_runs())

        if not check_runs:
            return "no_checks", 0, 0

        failed  = sum(1 for c in check_runs if c.conclusion in ("failure", "timed_out", "cancelled"))
        passed  = sum(1 for c in check_runs if c.conclusion == "success")
        overall = "failed" if failed > 0 else "passed"
        return overall, passed, failed
    except Exception:
        return "unknown", 0, 0


def collect_pr_data(token, username, max_repos, max_prs):
    g = Github(token)

    try:
        user = g.get_user(username)
    except Exception as e:
        print(f"  ERROR: Could not find user '{username}': {e}")
        return []

    repos = list(user.get_repos())
    print(f"  Found {len(repos)} repos. Scanning up to {max_repos}...\n")

    all_rows = []

    for repo in repos[:max_repos]:
        print(f"  Scanning: {repo.name}")
        try:
            prs = list(repo.get_pulls(state=PR_STATE)[:max_prs])
            print(f"    {len(prs)} PRs found")

            for pr in prs:
                ci_status, ci_passed, ci_failed = get_check_run_status(pr)

                # Time open in hours
                closed_at = pr.closed_at or datetime.now(timezone.utc)
                merged_at = pr.merged_at
                hours_open = round((closed_at - pr.created_at).total_seconds() / 3600, 1)

                row = {
                    "repo":            repo.name,
                    "pr_number":       pr.number,
                    "title":           pr.title,
                    "author":          pr.user.login if pr.user else "unknown",
                    "state":           pr.state,
                    "is_merged":       pr.merged if hasattr(pr, "merged") else False,
                    "created_at":      pr.created_at.strftime("%Y-%m-%d"),
                    "closed_at":       pr.closed_at.strftime("%Y-%m-%d") if pr.closed_at else None,
                    "hours_open":      hours_open,
                    "review_comments": pr.review_comments,
                    "total_comments":  pr.comments,
                    "commits":         pr.commits,
                    "changed_files":   pr.changed_files,
                    "additions":       pr.additions,
                    "deletions":       pr.deletions,
                    "reviewers":       ", ".join([r.login for r in pr.requested_reviewers]),
                    "ci_status":       ci_status,
                    "ci_checks_passed": ci_passed,
                    "ci_checks_failed": ci_failed,
                    "url":             pr.html_url,
                }
                all_rows.append(row)

        except Exception as e:
            print(f"    Skipping {repo.name}: {e}")

    return all_rows


def main():
    print("=" * 50)
    print("  GitHub PR Data Collector")
    print("=" * 50)

    if GITHUB_TOKEN == "ghp_your_token_here":
        print("\n  ERROR: Please set your GITHUB_TOKEN in the CONFIG section.\n")
        print("  How to get a token:")
        print("  1. Go to github.com > Settings > Developer Settings")
        print("  2. Personal access tokens > Tokens (classic)")
        print("  3. Generate new token with: repo (read), read:user scopes\n")
        return

    print(f"\n  Connecting to GitHub as '{GITHUB_USERNAME}'...")
    rows = collect_pr_data(GITHUB_TOKEN, GITHUB_USERNAME, MAX_REPOS, MAX_PRS_PER_REPO)

    if not rows:
        print("\n  No PR data collected. Check your token and username.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    # Quick summary
    print(f"\n{'=' * 50}")
    print(f"  Done! {len(df)} PRs collected from {df['repo'].nunique()} repos")
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"\n  Quick stats:")
    print(f"    Merged PRs    : {df['is_merged'].sum()}")
    print(f"    Closed (not merged): {((df['state'] == 'closed') & (~df['is_merged'])).sum()}")
    print(f"    CI failures   : {(df['ci_status'] == 'failed').sum()}")
    print(f"    Avg review comments per PR: {df['review_comments'].mean():.1f}")
    print(f"\n  Next step: run 02_analyse_with_ai.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
