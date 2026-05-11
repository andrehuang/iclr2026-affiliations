"""European-institution paper-count analysis for ICLR 2026.

Counting rule (matches the AI World NeurIPS leaderboard and the rest of this
repo): each institution is credited +1 per paper as long as at least one of
that paper's authors is affiliated with it, regardless of how many of the
paper's authors are listed under it. The same paper may therefore be counted
multiple times across distinct institutions.

This script reads the already-normalised, already-deduped-per-paper ranking
written by `make_iclr_treemap.py` (data/iclr2026_institutions_ranked_unique.csv)
and filters it to European institutions.

Outputs:
  - data/iclr2026_european_institutions.csv   (per-institution table)
  - data/iclr2026_european_by_country.csv     (per-country totals)
  - stdout                                    (formatted summary)

Run:
    python3 analyze_european_institutes.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SRC = DATA_DIR / "iclr2026_institutions_ranked_unique.csv"
OUT_INST = DATA_DIR / "iclr2026_european_institutions.csv"
OUT_COUNTRY = DATA_DIR / "iclr2026_european_by_country.csv"

TOTAL_PAPERS = 5356  # accepted ICLR 2026 papers, used for the %-of-corpus column

EUROPEAN_COUNTRIES = {
    "UK", "Switzerland", "Germany", "France",
    "Netherlands", "Belgium", "Sweden", "Finland", "Norway", "Denmark",
    "Italy", "Austria", "Czechia", "Spain", "Portugal", "Russia", "Ireland",
    "Poland", "Greece", "Hungary",
}


def load_european_rows() -> list[dict]:
    with SRC.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["country"] in EUROPEAN_COUNTRIES]


def write_institution_csv(rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: int(r["count"]), reverse=True)
    with OUT_INST.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["european_rank", "global_rank", "institution",
                    "papers", "pct_of_iclr2026", "country"])
        for i, r in enumerate(rows_sorted, 1):
            w.writerow([i, r["rank"], r["institution"], r["count"],
                        f"{int(r['count']) / TOTAL_PAPERS * 100:.2f}",
                        r["country"]])


def write_country_csv(rows: list[dict]) -> None:
    papers_by_country: dict[str, int] = defaultdict(int)
    insts_by_country: dict[str, int] = defaultdict(int)
    for r in rows:
        papers_by_country[r["country"]] += int(r["count"])
        insts_by_country[r["country"]] += 1
    ordered = sorted(papers_by_country.items(), key=lambda kv: kv[1], reverse=True)
    with OUT_COUNTRY.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["country", "institutions_in_dataset", "paper_affiliations"])
        for country, papers in ordered:
            w.writerow([country, insts_by_country[country], papers])


def print_summary(rows: list[dict]) -> None:
    total_affil = sum(int(r["count"]) for r in rows)
    print(f"European institutions in the dataset: {len(rows)}")
    print(f"Total paper–institution affiliations from Europe: {total_affil}")
    print(f"  (a single paper is counted once per distinct European institution)\n")

    papers_by_country: dict[str, int] = defaultdict(int)
    insts_by_country: dict[str, int] = defaultdict(int)
    for r in rows:
        papers_by_country[r["country"]] += int(r["count"])
        insts_by_country[r["country"]] += 1

    print("Paper affiliations by country (most → least):")
    print(f"{'Country':<14} {'Institutions':>12} {'Affiliations':>13}")
    for country, papers in sorted(papers_by_country.items(),
                                  key=lambda kv: kv[1], reverse=True):
        print(f"{country:<14} {insts_by_country[country]:>12} {papers:>13}")

    print("\nTop 30 European institutions by paper count:")
    print(f"{'#':>3}  {'GRank':>5}  {'Papers':>6}  {'%ICLR':>5}  Institution (Country)")
    rows_sorted = sorted(rows, key=lambda r: int(r["count"]), reverse=True)
    for i, r in enumerate(rows_sorted[:30], 1):
        pct = int(r["count"]) / TOTAL_PAPERS * 100
        print(f"{i:>3}  {r['rank']:>5}  {r['count']:>6}  {pct:>4.2f}%  "
              f"{r['institution']} ({r['country']})")


def main() -> None:
    rows = load_european_rows()
    print_summary(rows)
    write_institution_csv(rows)
    write_country_csv(rows)
    print(f"\nWrote {OUT_INST.relative_to(ROOT)}")
    print(f"Wrote {OUT_COUNTRY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
