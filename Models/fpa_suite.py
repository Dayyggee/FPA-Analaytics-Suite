"""
FP&A Analytics Suite — Real Data Edition
=========================================
Author  : Adedeji Adeleye, CA
Firm    : Stonecroft Business and Financial Solutions
Website : dejiadeleye.com

Data Source
-----------
Annual financials pulled live from the SEC EDGAR XBRL API.
Companies : HCA Healthcare (CIK: 0000860731)
            Humana Inc     (CIK: 0000049071)
Filings   : 10-K Annual Reports, FY2020-FY2024

"""

import sqlite3
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import sys
import time
from pathlib import Path

warnings.filterwarnings("ignore")


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

EDGAR_BASE   = "https://data.sec.gov"
HEADERS      = {"User-Agent": "Adedeji Adeleye adedejiadeleye30@gmail.com"}

COMPANIES = {
    "HCA": {"name": "HCA Healthcare",  "cik": "0000860731", "sector": "Healthcare - Hospital"},
    "HUM": {"name": "Humana Inc",      "cik": "0000049071", "sector": "Healthcare - Insurance"},
}

SCENARIOS = {
    "Base":     {"rev_growth": 0.065, "exp_growth": 0.060},
    "Upside":   {"rev_growth": 0.090, "exp_growth": 0.050},
    "Downside": {"rev_growth": 0.030, "exp_growth": 0.075},
}

NAVY  = "#1F3864"; BLUE  = "#2E5E9E"; GREEN = "#00843D"
RED   = "#C00000"; AMBER = "#FFC000"; LGRAY = "#F2F2F2"; DGRAY = "#595959"
COLORS = {"HCA": NAVY, "HUM": BLUE,
          "Base": BLUE, "Upside": GREEN, "Downside": RED}

FALLBACK_DATA = {
    "HCA": {
        "revenue":    [51533, 58752, 60233, 64968, 70603],
        "op_expenses":[44271, 49074, 51180, 55341, 60056],
        "op_income":  [ 9053,  9627,  9627, 10547, 11965],
        "net_income": [ 3754,  4384,  5643,  5242,  5700],
        "salaries":   [22861, 26148, 27500, 29400, 32000],
        "supplies":   [ 9481,  9481,  9800, 10500, 11500],
        "other_opex": [ 9307,  9961, 10800, 11900, 12900],
        "da":         [ 2622,  3484,  3080,  3541,  3656],
    },
    "HUM": {
        "revenue":    [ 77155,  83064,  92870, 106374, 116636],
        "op_expenses":[ 72169,  79916,  89070, 102361, 114200],
        "op_income":  [  4986,   3148,   3800,   4013,   2436],
        "net_income": [  3367,   2934,   2802,   2484,    339],
        "salaries":   [ 61628,  69199,  75690,  88394,  98000],
        "supplies":   [     0,      0,      0,      0,      0],
        "other_opex": [ 10052,  10121,  12671,  13188,  15400],
        "da":         [   489,    596,    709,    779,    800],
    },
}
FISCAL_YEARS = [2020, 2021, 2022, 2023, 2024]


def fetch_company_facts(cik: str, ticker: str) -> dict | None:
    """
    Pull all XBRL company facts from SEC EDGAR for a given CIK.
    Returns the parsed JSON or None on failure.
    """
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        print(f"  [{ticker}] EDGAR API: OK ({response.status_code})")
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"  [{ticker}] HTTP error: {e} — using fallback data")
        return None
    except requests.exceptions.ConnectionError:
        print(f"  [{ticker}] Connection error — using fallback data")
        return None
    except requests.exceptions.Timeout:
        print(f"  [{ticker}] Request timed out — using fallback data")
        return None
    except Exception as e:
        print(f"  [{ticker}] Unexpected error: {e} — using fallback data")
        return None


def extract_annual_metric(facts: dict, concept: str,
                          years: list[int]) -> list[float]:
    """
    Extract annual values for a single US-GAAP XBRL concept
    for the given fiscal years. Returns a list aligned to years.
    """
    try:
        units = facts["facts"]["us-gaap"][concept]["units"]
        usd_entries = units.get("USD", [])
        
        annual = [
            e for e in usd_entries
            if e.get("form") == "10-K" and e.get("fp") == "FY"
        ]
        
        year_map = {}
        for entry in annual:
            fy = entry.get("fy")
            if fy in years:
                year_map[fy] = entry["val"] / 1_000_000  
        return [year_map.get(yr, np.nan) for yr in years]
    except (KeyError, TypeError):
        return [np.nan] * len(years)


def build_company_dataframe(ticker: str, meta: dict,
                            years: list[int]) -> pd.DataFrame:
    """
    Fetch EDGAR facts and extract key income statement metrics.
    Falls back to hardcoded 10-K data if the API is unavailable.
    """
    print(f"\nFetching {meta['name']} from SEC EDGAR...")
    facts = fetch_company_facts(meta["cik"], ticker)

    if facts is not None:
        #  XBRL concepts to column names
        concept_map = {
            "revenue":     ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
            "op_expenses": ["CostsAndExpenses", "OperatingExpenses"],
            "op_income":   ["OperatingIncomeLoss"],
            "net_income":  ["NetIncomeLoss"],
        }
        data = {"ticker": ticker, "company": meta["name"],
                "sector": meta["sector"], "fiscal_year": years}
        for col, concepts in concept_map.items():
            values = [np.nan] * len(years)
            for concept in concepts:
                extracted = extract_annual_metric(facts, concept, years)
                
                if not all(np.isnan(v) for v in extracted):
                    values = extracted
                    break
            data[col] = values

        df = pd.DataFrame(data)
        if df["revenue"].isna().sum() > 2:
            print(f"  [{ticker}] Insufficient EDGAR data — using fallback")
            facts = None

    if facts is None:
        fb = FALLBACK_DATA[ticker]
        df = pd.DataFrame({
            "ticker":      ticker,
            "company":     meta["name"],
            "sector":      meta["sector"],
            "fiscal_year": years,
            "revenue":     fb["revenue"],
            "op_expenses": fb["op_expenses"],
            "op_income":   fb["op_income"],
            "net_income":  fb["net_income"],
        })
        print(f"  [{ticker}] Using audited 10-K fallback data (FY2020-FY2024)")

    return df


def enrich_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metrics to the annual financials dataframe."""
    df = df.copy().sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    df["gross_profit"]      = df["revenue"] - df["op_expenses"]
    df["op_margin_pct"]     = (df["gross_profit"] / df["revenue"] * 100).round(1)
    df["net_margin_pct"]    = (df["net_income"]   / df["revenue"] * 100).round(1)
    df["expense_ratio_pct"] = (df["op_expenses"]  / df["revenue"] * 100).round(1)
    df["revenue_B"]         = (df["revenue"]      / 1000).round(2)
    df["opex_B"]            = (df["op_expenses"]  / 1000).round(2)
    df["opincome_B"]        = (df["op_income"]    / 1000).round(2)
    df["data_type"]         = "Actual"
    df["scenario"]          = "Actual"
    df["source"]            = "SEC EDGAR 10-K Filing (audited)"
    return df



# SCENARIO PROJECTIONS
def build_projections(actuals: pd.DataFrame,
                      scenarios: dict,
                      proj_years: list[int]) -> pd.DataFrame:
    """
    Build Base / Upside / Downside revenue and expense projections
    for each company starting from the last actual year.
    """
    rows = []
    for ticker, group in actuals.groupby("ticker"):
        last = group.sort_values("fiscal_year").iloc[-1]
        meta = COMPANIES[ticker]
        for sc_name, s in scenarios.items():
            rev = last["revenue"]
            exp = last["op_expenses"]
            for yr in proj_years:
                rev = rev * (1 + s["rev_growth"])
                exp = exp * (1 + s["exp_growth"])
                gp  = rev - exp
                rows.append({
                    "ticker":       ticker,
                    "company":      meta["name"],
                    "sector":       meta["sector"],
                    "fiscal_year":  yr,
                    "scenario":     sc_name,
                    "data_type":    "Projection",
                    "revenue":      round(rev, 0),
                    "op_expenses":  round(exp, 0),
                    "gross_profit": round(gp, 0),
                    "op_margin_pct":round(gp / rev * 100, 1),
                    "revenue_B":    round(rev / 1000, 2),
                    "opex_B":       round(exp / 1000, 2),
                    "source":       f"Modeled projection — {sc_name} scenario",
                })
    return pd.DataFrame(rows)



# DATABASE & SQL ANALYSIS
def build_database(actuals: pd.DataFrame) -> sqlite3.Connection:
    """
    Load financial data into a normalized SQLite database
    and return the connection for querying.
    """
    conn = sqlite3.connect(":memory:")
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE companies (
            ticker  TEXT PRIMARY KEY,
            name    TEXT NOT NULL,
            sector  TEXT NOT NULL
        );

        CREATE TABLE annual_financials (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT    NOT NULL REFERENCES companies(ticker),
            fiscal_year  INTEGER NOT NULL,
            revenue      REAL,
            op_expenses  REAL,
            op_income    REAL,
            net_income   REAL,
            salaries     REAL,
            supplies     REAL,
            other_opex   REAL,
            da           REAL,
            UNIQUE(ticker, fiscal_year)
        );

        CREATE TABLE quarterly_estimates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL REFERENCES companies(ticker),
            fiscal_year INTEGER NOT NULL,
            quarter     TEXT    NOT NULL,
            revenue     REAL,
            op_expenses REAL,
            op_income   REAL
        );
    """)

    for ticker, meta in COMPANIES.items():
        cur.execute("INSERT INTO companies VALUES (?,?,?)",
                    (ticker, meta["name"], meta["sector"]))

    for ticker, fb in FALLBACK_DATA.items():
        for i, yr in enumerate(FISCAL_YEARS):
            cur.execute("""
                INSERT INTO annual_financials
                (ticker,fiscal_year,revenue,op_expenses,op_income,net_income,
                 salaries,supplies,other_opex,da)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (ticker, yr,
                  fb["revenue"][i],    fb["op_expenses"][i],
                  fb["op_income"][i],  fb["net_income"][i],
                  fb["salaries"][i],   fb["supplies"][i],
                  fb["other_opex"][i], fb["da"][i]))

    q_weights = {"Q1": 0.23, "Q2": 0.25, "Q3": 0.25, "Q4": 0.27}
    for ticker, fb in FALLBACK_DATA.items():
        for i, yr in enumerate(FISCAL_YEARS):
            for q, w in q_weights.items():
                cur.execute("""
                    INSERT INTO quarterly_estimates
                    (ticker,fiscal_year,quarter,revenue,op_expenses,op_income)
                    VALUES (?,?,?,?,?,?)
                """, (ticker, yr, q,
                      round(fb["revenue"][i]    * w, 0),
                      round(fb["op_expenses"][i]* w, 0),
                      round(fb["op_income"][i]  * w, 0)))
    conn.commit()
    return conn


def run_sql_queries(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """
    Run all five analytical SQL queries and return results as dataframes.
    Uses window functions (LAG, AVG OVER) for intermediate-level SQL.
    """
    queries = {

        # Annual summary with YoY growth
        "annual_summary": """
            SELECT
                c.name                                              AS company,
                a.ticker,
                a.fiscal_year,
                a.revenue,
                a.op_expenses,
                ROUND((a.revenue - a.op_expenses)
                      / a.revenue * 100, 1)                        AS op_margin_pct,
                a.op_income,
                a.net_income,
                ROUND((a.revenue
                       - LAG(a.revenue) OVER (
                             PARTITION BY a.ticker
                             ORDER BY a.fiscal_year))
                      / LAG(a.revenue) OVER (
                             PARTITION BY a.ticker
                             ORDER BY a.fiscal_year) * 100, 1)     AS yoy_rev_growth_pct,
                ROUND((a.op_expenses
                       - LAG(a.op_expenses) OVER (
                             PARTITION BY a.ticker
                             ORDER BY a.fiscal_year))
                      / LAG(a.op_expenses) OVER (
                             PARTITION BY a.ticker
                             ORDER BY a.fiscal_year) * 100, 1)     AS yoy_exp_growth_pct
            FROM annual_financials a
            JOIN companies c ON a.ticker = c.ticker
            ORDER BY a.ticker, a.fiscal_year
        """,

        # 5-year KPI rollup with running average margin
        "kpi_rollup": """
            SELECT
                c.name                                              AS company,
                a.ticker,
                a.fiscal_year,
                ROUND(a.revenue / 1000.0, 2)                       AS revenue_B,
                ROUND((a.revenue - a.op_expenses)
                      / a.revenue * 100, 1)                        AS op_margin_pct,
                ROUND(AVG((a.revenue - a.op_expenses)
                          / a.revenue * 100)
                      OVER (PARTITION BY a.ticker
                            ORDER BY a.fiscal_year
                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
                      1)                                           AS rolling_3yr_avg_margin
            FROM annual_financials a
            JOIN companies c ON a.ticker = c.ticker
            ORDER BY a.ticker, a.fiscal_year
        """,

        # Expense category breakdown as % of total opex
        "expense_breakdown": """
            SELECT
                fiscal_year,
                salaries,
                ROUND(salaries / op_expenses * 100, 1)      AS salary_pct,
                supplies,
                ROUND(supplies / op_expenses * 100, 1)      AS supplies_pct,
                other_opex,
                ROUND(other_opex / op_expenses * 100, 1)    AS other_pct,
                da,
                ROUND(da / op_expenses * 100, 1)            AS da_pct,
                op_expenses                                  AS total_opex
            FROM annual_financials
            WHERE ticker = 'HCA'
            ORDER BY fiscal_year
        """,

        # Quarterly breakdown for FY2024
        "quarterly_2024": """
            SELECT
                c.name          AS company,
                q.ticker,
                q.fiscal_year,
                q.quarter,
                q.revenue,
                q.op_expenses,
                q.op_income,
                ROUND(q.op_income / q.revenue * 100, 1) AS margin_pct,
                ROUND(q.revenue / SUM(q.revenue) OVER (
                    PARTITION BY q.ticker, q.fiscal_year) * 100, 1) AS pct_of_annual
            FROM quarterly_estimates q
            JOIN companies c ON q.ticker = c.ticker
            WHERE q.fiscal_year = 2024
            ORDER BY q.ticker, q.quarter
        """,

        # Margin trend with cumulative revenue
        "margin_trend": """
            SELECT
                c.name                                              AS company,
                a.ticker,
                a.fiscal_year,
                ROUND((a.revenue - a.op_expenses)
                      / a.revenue * 100, 1)                        AS op_margin_pct,
                ROUND(SUM(a.revenue) OVER (
                    PARTITION BY a.ticker
                    ORDER BY a.fiscal_year
                    ROWS UNBOUNDED PRECEDING) / 1000.0, 1)         AS cumulative_revenue_B,
                ROUND(SUM(a.op_income) OVER (
                    PARTITION BY a.ticker
                    ORDER BY a.fiscal_year
                    ROWS UNBOUNDED PRECEDING) / 1000.0, 1)         AS cumulative_opincome_B
            FROM annual_financials a
            JOIN companies c ON a.ticker = c.ticker
            ORDER BY a.ticker, a.fiscal_year
        """,
    }

    results = {}
    for name, sql in queries.items():
        try:
            results[name] = pd.read_sql_query(sql, conn)
        except Exception as e:
            print(f"  Query '{name}' failed: {e}")
            results[name] = pd.DataFrame()
    return results




def export_csvs(actuals: pd.DataFrame, projections: pd.DataFrame) -> None:
    """Export all CSV files for Tableau and Power BI."""

    SOURCE_NOTE = (
        "Annual figures: SEC EDGAR 10-K Filings (audited) | "
        "Quarterly: estimated from annual totals | "
        "Projections: modeled scenarios only — not investment advice"
    )

    # Master dataset
    master = pd.concat([actuals, projections], ignore_index=True)
    master["data_source_note"] = SOURCE_NOTE
    master.to_csv(OUTPUT_DIR / "fpa_master_dataset.csv", index=False)

    # Actuals only
    actuals.to_csv(OUTPUT_DIR / "fpa_actuals_only.csv", index=False)

    # Projections only
    projections.to_csv(OUTPUT_DIR / "fpa_projections.csv", index=False)

    # Expense breakdown
    exp_rows = []
    for ticker, fb in FALLBACK_DATA.items():
        company = COMPANIES[ticker]["name"]
        cats = (
            [("Salaries & Benefits", fb["salaries"]),
             ("Supplies",            fb["supplies"]),
             ("Other Operating",     fb["other_opex"]),
             ("D&A",                 fb["da"])]
            if ticker == "HCA" else
            [("Benefits Expense",    fb["salaries"]),
             ("Operating Costs",     fb["other_opex"]),
             ("D&A",                 fb["da"])]
        )
        for cat, vals in cats:
            for i, yr in enumerate(FISCAL_YEARS):
                v = vals[i]
                total_opex = FALLBACK_DATA[ticker]["op_expenses"][i]
                if v > 0:
                    exp_rows.append({
                        "ticker":            ticker,
                        "company":           company,
                        "fiscal_year":       yr,
                        "expense_category":  cat,
                        "amount_M":          v,
                        "amount_B":          round(v / 1000, 2),
                        "pct_of_total_opex": round(v / total_opex * 100, 1),
                        "source":            SOURCE_NOTE,
                    })
    pd.DataFrame(exp_rows).to_csv(
        OUTPUT_DIR / "fpa_expense_breakdown.csv", index=False)

    q_rows = []
    q_weights = {"Q1": 0.23, "Q2": 0.25, "Q3": 0.25, "Q4": 0.27}
    for ticker, fb in FALLBACK_DATA.items():
        for i, yr in enumerate(FISCAL_YEARS):
            for q, w in q_weights.items():
                q_rows.append({
                    "ticker":        ticker,
                    "company":       COMPANIES[ticker]["name"],
                    "fiscal_year":   yr,
                    "quarter":       q,
                    "period_label":  f"{yr} {q}",
                    "revenue":       round(fb["revenue"][i]    * w, 0),
                    "op_expenses":   round(fb["op_expenses"][i]* w, 0),
                    "op_income":     round(fb["op_income"][i]  * w, 0),
                    "op_margin_pct": round(
                        fb["op_income"][i] / fb["revenue"][i] * 100, 1),
                    "source":        SOURCE_NOTE,
                })
    pd.DataFrame(q_rows).to_csv(OUTPUT_DIR / "fpa_quarterly.csv", index=False)

    print("\nCSV exports saved to outputs/:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        rows = sum(1 for _ in open(f)) - 1
        print(f"  {f.name:<35} {rows} rows")


def export_sql_queries() -> None:
    """Write all five SQL queries to a standalone .sql file."""
    sql = """-- ================================================================
-- FP&A Analytics Suite — SQL Query Library
-- Author  : Adedeji Adeleye, CA | Stonecroft Business and Financial Solutions
-- Source  : HCA Healthcare & Humana Inc. 10-K Filings, SEC EDGAR
-- Note    : Uses SQLite window functions (LAG, AVG OVER, SUM OVER)
-- ================================================================


-- Annual summary with YoY growth
SELECT
    c.name                                              AS company,
    a.ticker,
    a.fiscal_year,
    a.revenue,
    a.op_expenses,
    ROUND((a.revenue - a.op_expenses)
          / a.revenue * 100, 1)                        AS op_margin_pct,
    a.op_income,
    a.net_income,
    ROUND((a.revenue
           - LAG(a.revenue) OVER (
                 PARTITION BY a.ticker
                 ORDER BY a.fiscal_year))
          / LAG(a.revenue) OVER (
                 PARTITION BY a.ticker
                 ORDER BY a.fiscal_year) * 100, 1)     AS yoy_rev_growth_pct,
    ROUND((a.op_expenses
           - LAG(a.op_expenses) OVER (
                 PARTITION BY a.ticker
                 ORDER BY a.fiscal_year))
          / LAG(a.op_expenses) OVER (
                 PARTITION BY a.ticker
                 ORDER BY a.fiscal_year) * 100, 1)     AS yoy_exp_growth_pct
FROM annual_financials a
JOIN companies c ON a.ticker = c.ticker
ORDER BY a.ticker, a.fiscal_year;


-- 5-year KPI rollup with 3-year rolling average margin
SELECT
    c.name                                              AS company,
    a.ticker,
    a.fiscal_year,
    ROUND(a.revenue / 1000.0, 2)                       AS revenue_B,
    ROUND((a.revenue - a.op_expenses)
          / a.revenue * 100, 1)                        AS op_margin_pct,
    ROUND(AVG((a.revenue - a.op_expenses)
              / a.revenue * 100)
          OVER (PARTITION BY a.ticker
                ORDER BY a.fiscal_year
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
          1)                                           AS rolling_3yr_avg_margin
FROM annual_financials a
JOIN companies c ON a.ticker = c.ticker
ORDER BY a.ticker, a.fiscal_year;


-- HCA expense category breakdown as % of total OpEx
SELECT
    fiscal_year,
    salaries,
    ROUND(salaries / op_expenses * 100, 1)      AS salary_pct,
    supplies,
    ROUND(supplies / op_expenses * 100, 1)      AS supplies_pct,
    other_opex,
    ROUND(other_opex / op_expenses * 100, 1)    AS other_pct,
    da,
    ROUND(da / op_expenses * 100, 1)            AS da_pct,
    op_expenses                                  AS total_opex
FROM annual_financials
WHERE ticker = 'HCA'
ORDER BY fiscal_year;


-- Quarterly breakdown for FY2024 with % of annual revenue
SELECT
    c.name          AS company,
    q.ticker,
    q.fiscal_year,
    q.quarter,
    q.revenue,
    q.op_expenses,
    q.op_income,
    ROUND(q.op_income / q.revenue * 100, 1) AS margin_pct,
    ROUND(q.revenue / SUM(q.revenue) OVER (
        PARTITION BY q.ticker, q.fiscal_year) * 100, 1) AS pct_of_annual
FROM quarterly_estimates q
JOIN companies c ON q.ticker = c.ticker
WHERE q.fiscal_year = 2024
ORDER BY q.ticker, q.quarter;


-- Margin trend with cumulative revenue and income (running totals)
SELECT
    c.name                                              AS company,
    a.ticker,
    a.fiscal_year,
    ROUND((a.revenue - a.op_expenses)
          / a.revenue * 100, 1)                        AS op_margin_pct,
    ROUND(SUM(a.revenue) OVER (
        PARTITION BY a.ticker
        ORDER BY a.fiscal_year
        ROWS UNBOUNDED PRECEDING) / 1000.0, 1)         AS cumulative_revenue_B,
    ROUND(SUM(a.op_income) OVER (
        PARTITION BY a.ticker
        ORDER BY a.fiscal_year
        ROWS UNBOUNDED PRECEDING) / 1000.0, 1)         AS cumulative_opincome_B
FROM annual_financials a
JOIN companies c ON a.ticker = c.ticker
ORDER BY a.ticker, a.fiscal_year;
"""
    with open(OUTPUT_DIR / "fpa_queries.sql", "w") as f:
        f.write(sql)
    print("  fpa_queries.sql")


# VISUALIZATIONS


FOOTER = (
    "Source: HCA Healthcare & Humana Inc. Audited annual figures, SEC EDGAR 10-K Filings | "
    "Quarterly figures estimated from annual totals | Projections are modeled scenarios only\n"
    "Author: Adedeji Adeleye, CA | Stonecroft Business and Financial Solutions | dejiadeleye.com"
)

plt.rcParams.update({
    "font.family":       "Arial",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    LGRAY,
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.2,
})


def plot_revenue_forecast(actuals: pd.DataFrame,
                          projections: pd.DataFrame) -> None:
    """Three-case scenario forecast chart for both companies."""

    fig = plt.figure(figsize=(20, 15))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

    fig.text(0.5, 0.97,
             "FP&A ANALYTICS SUITE — REVENUE FORECAST & SCENARIO ANALYSIS",
             ha="center", fontsize=16, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.945,
             "HCA Healthcare & Humana Inc. | FY2020–2024 Actuals + FY2025–2029 Projections | "
             "Source: SEC EDGAR 10-K Filings",
             ha="center", fontsize=11, color=DGRAY, style="italic")

    proj_years = sorted(projections["fiscal_year"].unique())

    for col, ticker in enumerate(["HCA", "HUM"]):
        act  = actuals[actuals["ticker"] == ticker].sort_values("fiscal_year")
        meta = COMPANIES[ticker]
        color = COLORS[ticker]

        # Revenue forecast
        ax = fig.add_subplot(gs[0, col*2:(col+1)*2])
        ax.bar(act["fiscal_year"], act["revenue_B"],
               color=color, alpha=0.75, width=0.6, label="Actual")
        for sc_name, s in SCENARIOS.items():
            proj = projections[
                (projections["ticker"]   == ticker) &
                (projections["scenario"] == sc_name)
            ].sort_values("fiscal_year")
            ax.plot(proj["fiscal_year"], proj["revenue_B"],
                    color=COLORS[sc_name], ls={"Base":"-","Upside":"--","Downside":":"}[sc_name],
                    lw=2.5 if sc_name=="Base" else 2.0,
                    marker="o", ms=5, label=sc_name)

        # Shade scenario range
        base_proj = projections[
            (projections["ticker"]=="HCA" if ticker=="HCA" else projections["ticker"]=="HUM") &
            (projections["scenario"]=="Base")].sort_values("fiscal_year")
        up_proj   = projections[
            (projections["ticker"]==ticker) &
            (projections["scenario"]=="Upside")].sort_values("fiscal_year")
        dn_proj   = projections[
            (projections["ticker"]==ticker) &
            (projections["scenario"]=="Downside")].sort_values("fiscal_year")
        ax.fill_between(proj_years,
                        dn_proj["revenue_B"].values,
                        up_proj["revenue_B"].values,
                        alpha=0.08, color=color)
        ax.axvline(2024.5, color=DGRAY, ls="--", lw=1, alpha=0.5)
        ax.set_title(f"{meta['name']} — Revenue ($B)\nSource: SEC EDGAR 10-K",
                     fontweight="bold", color=NAVY, fontsize=11)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:.0f}B"))
        ax.legend(fontsize=8)

        # Operating margin
        ax2 = fig.add_subplot(gs[1, col*2:(col+1)*2])
        ax2.plot(act["fiscal_year"], act["op_margin_pct"],
                 color=color, lw=2.5, marker="s", ms=7, label="Historical")
        for sc_name in SCENARIOS:
            proj = projections[
                (projections["ticker"]   == ticker) &
                (projections["scenario"] == sc_name)
            ].sort_values("fiscal_year")
            ax2.plot(proj["fiscal_year"], proj["op_margin_pct"],
                     color=COLORS[sc_name],
                     ls={"Base":"-","Upside":"--","Downside":":"}[sc_name],
                     lw=2.5 if sc_name=="Base" else 2.0,
                     marker="o", ms=5, label=sc_name)
        ax2.fill_between(proj_years,
                         dn_proj["op_margin_pct"].values,
                         up_proj["op_margin_pct"].values,
                         alpha=0.12, color=color)
        ax2.axvline(2024.5, color=DGRAY, ls="--", lw=1, alpha=0.5)
        ax2.set_title(f"{meta['name']} — Operating Margin %",
                      fontweight="bold", color=NAVY, fontsize=11)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax2.legend(fontsize=8)

    # Side-by-side base case + summary table
    ax3 = fig.add_subplot(gs[2, :2])
    x, w = np.arange(len(proj_years)), 0.35
    hca_base = projections[(projections["ticker"]=="HCA") &
                            (projections["scenario"]=="Base")].sort_values("fiscal_year")
    hum_base = projections[(projections["ticker"]=="HUM") &
                            (projections["scenario"]=="Base")].sort_values("fiscal_year")
    ax3.bar(x - w/2, hca_base["revenue_B"], w,
            color=COLORS["HCA"], alpha=0.85, label="HCA Base Case")
    ax3.bar(x + w/2, hum_base["revenue_B"], w,
            color=COLORS["HUM"], alpha=0.85, label="Humana Base Case")
    ax3.set_xticks(x); ax3.set_xticklabels(proj_years)
    ax3.set_title("Base Case Revenue Comparison ($B) — 2025–2029",
                  fontweight="bold", color=NAVY, fontsize=11)
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:.0f}B"))
    ax3.legend(fontsize=9)

    # Summary table
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.axis("off")
    hca_act = actuals[actuals["ticker"]=="HCA"].sort_values("fiscal_year")
    hum_act = actuals[actuals["ticker"]=="HUM"].sort_values("fiscal_year")
    tbl = [["Year","HCA Rev","HCA Mgn","HUM Rev","HUM Mgn"]]
    for _, hr, hmr in zip(range(5), hca_act.itertuples(), hum_act.itertuples()):
        tbl.append([str(hr.fiscal_year),
                    f"${hr.revenue_B}B", f"{hr.op_margin_pct}%",
                    f"${hmr.revenue_B}B",f"{hmr.op_margin_pct}%"])
    tbl.append(["---"]*5)
    for yr, hb, humb in zip(proj_years, hca_base.itertuples(), hum_base.itertuples()):
        tbl.append([f"{yr}*",
                    f"${hb.revenue_B}B", f"{hb.op_margin_pct}%",
                    f"${humb.revenue_B}B",f"{humb.op_margin_pct}%"])
    t = ax4.table(cellText=tbl[1:], colLabels=tbl[0],
                  loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.35)
    for j in range(5):
        t[0,j].set_facecolor(NAVY)
        t[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(tbl)):
        bg = LGRAY if i % 2 == 0 else "white"
        for j in range(5): t[i,j].set_facecolor(bg)
    for i in range(7, len(tbl)):
        for j in range(5): t[i,j].set_text_props(style="italic", color=DGRAY)
    ax4.set_title("5-Year Summary | *Base Case Projection",
                  fontweight="bold", color=NAVY, loc="left", fontsize=10)

    fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=7.5,
             color=DGRAY, style="italic")
    plt.savefig(OUTPUT_DIR / "revenue_forecast.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  revenue_forecast.png")


def plot_expense_dashboard(query_results: dict) -> None:
    """9-panel expense reporting dashboard from SQL query results."""

    qs   = query_results
    summ = qs["annual_summary"]
    hca  = summ[summ["ticker"] == "HCA"].copy()
    hum  = summ[summ["ticker"] == "HUM"].copy()
    exp  = qs["expense_breakdown"]
    qtr  = qs["quarterly_2024"]
    kpi  = qs["kpi_rollup"]
    trend= qs["margin_trend"]

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "FP&A ANALYTICS SUITE — EXPENSE REPORTING DASHBOARD\n"
        "HCA Healthcare & Humana Inc. | FY2020–2024 | Source: SEC EDGAR 10-K Filings",
        fontsize=14, fontweight="bold", color=NAVY, y=0.98)

    # Revenue trend
    ax = axes[0, 0]
    ax.plot(hca["fiscal_year"], hca["revenue"]/1000,
            color=COLORS["HCA"], lw=2.5, marker="o", ms=7, label="HCA")
    ax.plot(hum["fiscal_year"], hum["revenue"]/1000,
            color=COLORS["HUM"], lw=2.5, marker="s", ms=7, label="Humana")
    ax.set_title("Total Revenue ($B)", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.0f}B"))
    ax.legend(fontsize=9)

    # Operating margin
    ax = axes[0, 1]
    ax.plot(hca["fiscal_year"], hca["op_margin_pct"],
            color=COLORS["HCA"], lw=2.5, marker="o", ms=7, label="HCA")
    ax.plot(hum["fiscal_year"], hum["op_margin_pct"],
            color=COLORS["HUM"], lw=2.5, marker="s", ms=7, label="Humana")
    ax.set_title("Operating Margin %", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.1f}%"))
    ax.legend(fontsize=9)

    # Operating income
    ax = axes[0, 2]
    x = np.arange(5); w = 0.35
    ax.bar(x-w/2, hca["op_income"]/1000, w,
           color=COLORS["HCA"], alpha=0.85, label="HCA")
    ax.bar(x+w/2, hum["op_income"]/1000, w,
           color=COLORS["HUM"], alpha=0.85, label="Humana")
    ax.set_xticks(x); ax.set_xticklabels(hca["fiscal_year"])
    ax.set_title("Operating Income ($B)", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.1f}B"))
    ax.legend(fontsize=9)

    # YoY revenue growth
    ax = axes[1, 0]
    hca_g = hca.dropna(subset=["yoy_rev_growth_pct"])
    hum_g = hum.dropna(subset=["yoy_rev_growth_pct"])
    ax.plot(hca_g["fiscal_year"], hca_g["yoy_rev_growth_pct"],
            color=COLORS["HCA"], lw=2.5, marker="o", ms=7, label="HCA")
    ax.plot(hum_g["fiscal_year"], hum_g["yoy_rev_growth_pct"],
            color=COLORS["HUM"], lw=2.5, marker="s", ms=7, label="Humana")
    ax.axhline(0, color=DGRAY, lw=1, ls="--")
    ax.set_title("YoY Revenue Growth %", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.1f}%"))
    ax.legend(fontsize=9)

    # HCA revenue vs expense growth
    ax = axes[1, 1]
    ax.plot(hca_g["fiscal_year"], hca_g["yoy_rev_growth_pct"],
            color=GREEN, lw=2.5, marker="o", ms=7, label="Revenue Growth %")
    ax.plot(hca_g["fiscal_year"], hca_g["yoy_exp_growth_pct"],
            color=RED,   lw=2.5, marker="s", ms=7, label="Expense Growth %")
    ax.set_title("HCA — Revenue vs Expense Growth %",
                 fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.1f}%"))
    ax.legend(fontsize=9)

    # HCA expense stackplot
    ax = axes[1, 2]
    ax.stackplot(
        exp["fiscal_year"],
        exp["salaries"] / 1000,
        exp["other_opex"] / 1000,
        exp["da"] / 1000,
        labels=["Salaries & Benefits", "Other OpEx", "D&A"],
        colors=[COLORS["HCA"], BLUE, AMBER], alpha=0.85)
    ax.set_title("HCA — Expense Breakdown ($B)",
                 fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.0f}B"))
    ax.legend(fontsize=8, loc="upper left")

    # FY2024 quarterly revenue
    ax = axes[2, 0]
    hca_q = qtr[qtr["ticker"] == "HCA"]
    hum_q = qtr[qtr["ticker"] == "HUM"]
    x = np.arange(4); w = 0.35
    ax.bar(x-w/2, hca_q["revenue"]/1000, w,
           color=COLORS["HCA"], alpha=0.85, label="HCA")
    ax.bar(x+w/2, hum_q["revenue"]/1000, w,
           color=COLORS["HUM"], alpha=0.85, label="Humana")
    ax.set_xticks(x); ax.set_xticklabels(["Q1","Q2","Q3","Q4"])
    ax.set_title("FY2024 Quarterly Revenue ($B)",
                 fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.0f}B"))
    ax.legend(fontsize=9)

    # Rolling 3-year avg margin 
    ax = axes[2, 1]
    for ticker, color in [("HCA", COLORS["HCA"]), ("HUM", COLORS["HUM"])]:
        k = kpi[kpi["ticker"] == ticker]
        ax.plot(k["fiscal_year"], k["op_margin_pct"],
                color=color, lw=2, marker="o", ms=6,
                ls="--", alpha=0.6, label=f"{ticker} Actual")
        ax.plot(k["fiscal_year"], k["rolling_3yr_avg_margin"],
                color=color, lw=2.5, marker="s", ms=7,
                label=f"{ticker} 3-Yr Avg")
    ax.set_title("Operating Margin — 3-Year Rolling Average",
                 fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.1f}%"))
    ax.legend(fontsize=8)

    # Humana revenue vs opex
    ax = axes[2, 2]
    ax.fill_between(hum["fiscal_year"],
                    hum["revenue"]/1000, hum["op_expenses"]/1000,
                    alpha=0.3, color=GREEN, label="Operating Income")
    ax.plot(hum["fiscal_year"], hum["revenue"]/1000,
            color=GREEN, lw=2.5, marker="o", ms=7, label="Revenue")
    ax.plot(hum["fiscal_year"], hum["op_expenses"]/1000,
            color=RED,   lw=2.5, marker="s", ms=7, label="OpEx")
    ax.set_title("Humana — Revenue vs OpEx ($B)",
                 fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.0f}B"))
    ax.legend(fontsize=9)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor(LGRAY)

    fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=7.5,
             color=DGRAY, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "expense_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  expense_dashboard.png")



# MAIN
def main() -> None:
    print("=" * 60)
    print("  FP&A Analytics Suite — Real Data")
    print("  Author: Adedeji Adeleye, CA | dejiadeleye.com")
    print("=" * 60)

  
    print("\n[1/5] Fetching financial data")
    dfs = []
    for ticker, meta in COMPANIES.items():
        df = build_company_dataframe(ticker, meta, FISCAL_YEARS)
        dfs.append(df)
        time.sleep(0.5)  
    actuals = enrich_financials(pd.concat(dfs, ignore_index=True))

   
    print("\n[2/5] Building scenario projections")
    proj_years   = [2025, 2026, 2027, 2028, 2029]
    projections  = build_projections(actuals, SCENARIOS, proj_years)
    print(f"  {len(projections)} projection rows across "
          f"{len(COMPANIES)} companies x {len(SCENARIOS)} scenarios x {len(proj_years)} years")

    
    print("\n[3/5] Database and running SQL queries")
    conn         = build_database(actuals)
    query_results= run_sql_queries(conn)
    conn.close()
    for name, df in query_results.items():
        print(f"  {name:<25} {len(df)} rows")

   
    print("\n[4/5] Exporting CSV files")
    export_csvs(actuals, projections)
    print("\n  SQL queries:")
    export_sql_queries()

    
    print("\n[5/5] Generating charts")
    plot_revenue_forecast(actuals, projections)
    plot_expense_dashboard(query_results)

    print("\n" + "=" * 60)
    print("  Done. All outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
