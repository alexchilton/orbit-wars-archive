from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright


def _dismiss_cookie_banner(page) -> None:
    button = page.locator("button:has-text('OK, Got it.')")
    if button.count() > 0:
        try:
            button.first.click(timeout=2000)
            page.wait_for_timeout(250)
        except Exception:
            pass


def _extract_submission_rows(page, base_url: str, limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen_ids: set[str] = set()
    page.wait_for_selector("span[title*='highest scoring agent']", timeout=30000)
    _dismiss_cookie_banner(page)
    count = page.locator("span[title*='highest scoring agent']").count()

    for index in range(count):
        if index > 0:
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("span[title*='highest scoring agent']", timeout=30000)
            _dismiss_cookie_banner(page)
        button = page.locator("span[title*='highest scoring agent']").nth(index)
        row = button.locator("xpath=ancestor::li[1]")
        row_text = (row.inner_text(timeout=1000) or "").strip()
        button.click()
        page.wait_for_timeout(600)
        current_url = page.url
        query = parse_qs(urlparse(current_url).query)
        submission_id = next(iter(query.get("submissionId", [])), "")
        episode_id = next(iter(query.get("episodeId", [])), "")
        if not submission_id or submission_id in seen_ids:
            continue
        seen_ids.add(submission_id)

        label = (button.inner_text(timeout=1000) or "").strip()
        team_name = ""
        lines = [line.strip() for line in row_text.splitlines() if line.strip()]
        if len(lines) >= 2:
            team_name = lines[1]

        rows.append(
            {
                "submission_id": submission_id,
                "episode_id": episode_id,
                "href": current_url,
                "label": label,
                "team_name": team_name,
                "row_text": row_text,
            }
        )
        if len(rows) >= limit:
            break

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Kaggle leaderboard submission IDs from rendered leaderboard links")
    parser.add_argument("--competition", default="orbit-wars")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")
    parser.add_argument("--timeout", type=int, default=45000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leaderboard_url = f"https://www.kaggle.com/competitions/{args.competition}/leaderboard"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        page = browser.new_page()
        page.set_default_timeout(args.timeout)
        page.goto(leaderboard_url, wait_until="networkidle")
        rows = _extract_submission_rows(page, leaderboard_url, args.limit)
        browser.close()

    payload = {
        "competition": args.competition,
        "leaderboard_url": leaderboard_url,
        "count": len(rows),
        "rows": rows,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
