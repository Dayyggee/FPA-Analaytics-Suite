"""
FP&A Analytics Suite — Insurance Models
=========================================
Companies : Lincoln National Corporation (NYSE: LNC)
            Unum Group (NYSE: UNM)
Source    : SEC EDGAR 10-K Annual Filings, FY2020-FY2024
Author    : Adedeji Adeleye , CA | Stonecroft Business and Financial Solutions
Website   : dejiadeleye.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sqlite3
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(r"C:\Users\adele\Desktop\Resume\FPA-Analytics-Suite\Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

NAVY  = "#1F3864"; BLUE  = "#2E5E9E"; GREEN = "#00843D"
RED   = "#C00000"; AMBER = "#FFC000"; LGRAY = "#F2F2F2"; DGRAY = "#595959"
LNC_COLOR = "#1F3864"; UNM_COLOR = "#2E5E9E"

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

FOOTER = (
    "Source: Lincoln National & Unum Group Audited annual figures, SEC EDGAR 10-K Filings | "
    "Quarterly figures estimated from annual totals | Projections are modeled scenarios only\n"
    "Author: Adedeji Adeleye, CA | Stonecroft Business and Financial Solutions | dejiadeleye.com"
)

COMPANIES_MAP = {"LNC": "Lincoln National", "UNM": "Unum Group"}

#  Data from SEC EDGAR 10-K Filings ($ millions)

# Lincoln National Corporation (NYSE: LNC)
# Source: LNC 10-K Filings FY2020-FY2024, SEC EDGAR
lnc = pd.DataFrame({
    "year":       [2020,  2021,  2022,  2023,  2024],
    "revenue":    [17213, 19029, 16617, 15912, 17521],
    "op_expenses":[15821, 17069, 19975, 14778, 15643],
    "op_income":  [1392,  1960,  -3358, 1134,  1878],
    "net_income": [975,   1613,  -2559, 857,   1426],
    "benefits":   [10901, 11832, 14698, 9124,  9872],
    "commissions":[2819,  2781,  2801,  2761,  2836],
    "other_opex": [2101,  2456,  2476,  2893,  2935],
})
lnc["company"] = "Lincoln National"
lnc["ticker"]  = "LNC"
lnc["sector"]  = "Life & Annuity Insurance"

# Unum Group (NYSE: UNM)
# Source: UNM 10-K Filings / Yahoo Finance FY2020-FY2024, SEC EDGAR
unm = pd.DataFrame({
    "year":       [2020,  2021,  2022,  2023,  2024],
    "revenue":    [11846, 11838, 11875, 12309, 12793],
    "op_expenses":[10812, 10577, 10125, 10669, 10541],
    "op_income":  [1034,  1261,  1750,  1640,  2252],
    "net_income": [730,   981,   1407,  1284,  1779],
    "benefits":   [7825,  7598,  7012,  7680,  7420],
    "commissions":[1057,  1057,  1082,  1139,  1188],
    "other_opex": [1930,  1922,  2031,  1850,  1933],
})
unm["company"] = "Unum Group"
unm["ticker"]  = "UNM"
unm["sector"]  = "Group Benefits Insurance"

FISCAL_YEARS = [2020, 2021, 2022, 2023, 2024]

def enrich(df):
    """Add derived metrics to the annual financials dataframe."""
    df = df.copy().sort_values("year").reset_index(drop=True)
    df["gross_profit"]   = df["revenue"] - df["op_expenses"]
    df["op_margin_pct"]  = (df["gross_profit"] / df["revenue"] * 100).round(1)
    df["net_margin_pct"] = (df["net_income"]   / df["revenue"] * 100).round(1)
    df["expense_ratio"]  = (df["op_expenses"]  / df["revenue"] * 100).round(1)
    df["revenue_B"]      = (df["revenue"]      / 1000).round(2)
    df["opex_B"]         = (df["op_expenses"]  / 1000).round(2)
    df["yoy_rev_growth"] = df["revenue"].pct_change().mul(100).round(1)
    df["yoy_exp_growth"] = df["op_expenses"].pct_change().mul(100).round(1)
    df["data_type"]      = "Actual"
    df["scenario"]       = "Actual"
    df["source"]         = "SEC EDGAR 10-K Filing (audited)"
    return df

lnc     = enrich(lnc)
unm     = enrich(unm)
actuals = pd.concat([lnc, unm], ignore_index=True)


# SCENARIO PROJECTIONS

scenarios  = {
    "Base":     {"rg": 0.055, "eg": 0.050},
    "Upside":   {"rg": 0.080, "eg": 0.040},
    "Downside": {"rg": 0.020, "eg": 0.065},
}
proj_years = [2025, 2026, 2027, 2028, 2029]
sc_colors  = {"Base": BLUE, "Upside": GREEN, "Downside": RED}
sc_ls      = {"Base": "-",  "Upside": "--",  "Downside": ":"}

def build_projections(df_list, scenarios, proj_years):
    """Build Base/Upside/Downside projections for each company."""
    rows = []
    for df, ticker, company in df_list:
        last = df.sort_values("year").iloc[-1]
        for sc_name, s in scenarios.items():
            rev = last["revenue"]
            exp = last["op_expenses"]
            for yr in proj_years:
                rev = rev * (1 + s["rg"])
                exp = exp * (1 + s["eg"])
                gp  = rev - exp
                rows.append({
                    "ticker":       ticker,
                    "company":      company,
                    "year":         yr,
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

proj = build_projections(
    [(lnc, "LNC", "Lincoln National"), (unm, "UNM", "Unum Group")],
    scenarios, proj_years
)



# DATABASE & SQL QUERIES

def build_database(lnc_df, unm_df):
    """Load financials into a normalized SQLite database."""
    conn = sqlite3.connect(":memory:")
    cur  = conn.cursor()
    cur.executescript("""
        CREATE TABLE companies (
            ticker TEXT PRIMARY KEY,
            name   TEXT NOT NULL,
            sector TEXT NOT NULL
        );
        CREATE TABLE financials (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT    NOT NULL REFERENCES companies(ticker),
            fiscal_year  INTEGER NOT NULL,
            revenue      REAL, op_expenses REAL, op_income REAL, net_income REAL,
            benefits     REAL, commissions REAL, other_opex  REAL
        );
        CREATE TABLE quarterly (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL REFERENCES companies(ticker),
            fiscal_year INTEGER NOT NULL,
            quarter     TEXT    NOT NULL,
            revenue     REAL, op_expenses REAL, op_income REAL
        );
    """)
    cur.executemany("INSERT INTO companies VALUES (?,?,?)", [
        ("LNC", "Lincoln National", "Life & Annuity Insurance"),
        ("UNM", "Unum Group",       "Group Benefits Insurance"),
    ])
    for ticker, df_e in [("LNC", lnc_df), ("UNM", unm_df)]:
        for _, r in df_e.iterrows():
            cur.execute("""
                INSERT INTO financials
                (ticker,fiscal_year,revenue,op_expenses,op_income,net_income,
                 benefits,commissions,other_opex)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (ticker, int(r["year"]), r["revenue"], r["op_expenses"],
                  r["op_income"], r["net_income"],
                  r["benefits"], r["commissions"], r["other_opex"]))
    q_weights = {"Q1": 0.23, "Q2": 0.25, "Q3": 0.25, "Q4": 0.27}
    for ticker, df_e in [("LNC", lnc_df), ("UNM", unm_df)]:
        for _, r in df_e.iterrows():
            for q, w in q_weights.items():
                cur.execute("""
                    INSERT INTO quarterly
                    (ticker,fiscal_year,quarter,revenue,op_expenses,op_income)
                    VALUES (?,?,?,?,?,?)
                """, (ticker, int(r["year"]), q,
                      round(r["revenue"]    * w, 0),
                      round(r["op_expenses"]* w, 0),
                      round(r["op_income"]  * w, 0)))
    conn.commit()
    return conn


def run_queries(conn):
    """Run all five analytical SQL queries using window functions."""
    queries = {

        "annual_yoy": """
            SELECT c.name, a.ticker, a.fiscal_year,
                a.revenue, a.op_expenses,
                ROUND((a.revenue - a.op_expenses) / a.revenue * 100, 1) AS op_margin_pct,
                a.op_income, a.net_income,
                ROUND((a.revenue - LAG(a.revenue) OVER (
                    PARTITION BY a.ticker ORDER BY a.fiscal_year))
                    / LAG(a.revenue) OVER (
                    PARTITION BY a.ticker ORDER BY a.fiscal_year) * 100, 1) AS yoy_rev_growth,
                ROUND((a.op_expenses - LAG(a.op_expenses) OVER (
                    PARTITION BY a.ticker ORDER BY a.fiscal_year))
                    / LAG(a.op_expenses) OVER (
                    PARTITION BY a.ticker ORDER BY a.fiscal_year) * 100, 1) AS yoy_exp_growth
            FROM financials a
            JOIN companies c ON a.ticker = c.ticker
            ORDER BY a.ticker, a.fiscal_year
        """,

        "rolling_margin": """
            SELECT c.name, a.ticker, a.fiscal_year,
                ROUND((a.revenue - a.op_expenses) / a.revenue * 100, 1) AS op_margin_pct,
                ROUND(AVG((a.revenue - a.op_expenses) / a.revenue * 100) OVER (
                    PARTITION BY a.ticker ORDER BY a.fiscal_year
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1) AS rolling_3yr_margin,
                ROUND(SUM(a.revenue) OVER (
                    PARTITION BY a.ticker ORDER BY a.fiscal_year
                    ROWS UNBOUNDED PRECEDING) / 1000.0, 1) AS cumulative_rev_B
            FROM financials a
            JOIN companies c ON a.ticker = c.ticker
            ORDER BY a.ticker, a.fiscal_year
        """,

        "expense_breakdown": """
            SELECT fiscal_year,
                benefits,    ROUND(benefits    / op_expenses * 100, 1) AS benefits_pct,
                commissions, ROUND(commissions / op_expenses * 100, 1) AS comm_pct,
                other_opex,  ROUND(other_opex  / op_expenses * 100, 1) AS other_pct,
                op_expenses AS total_opex
            FROM financials
            WHERE ticker = 'UNM'
            ORDER BY fiscal_year
        """,

        "quarterly_2024": """
            SELECT c.name, q.ticker, q.fiscal_year, q.quarter,
                q.revenue, q.op_expenses, q.op_income,
                ROUND(q.op_income / q.revenue * 100, 1) AS margin_pct,
                ROUND(q.revenue / SUM(q.revenue) OVER (
                    PARTITION BY q.ticker, q.fiscal_year) * 100, 1) AS pct_of_annual
            FROM quarterly q
            JOIN companies c ON q.ticker = c.ticker
            WHERE q.fiscal_year = 2024
            ORDER BY q.ticker, q.quarter
        """,

        "kpi_summary": """
            SELECT c.name, a.ticker,
                ROUND(SUM(a.revenue)    / 1000.0, 1) AS total_rev_5yr_B,
                ROUND(AVG((a.revenue - a.op_expenses) / a.revenue * 100), 1) AS avg_margin_5yr,
                ROUND((MAX(a.revenue) - MIN(a.revenue)) / MIN(a.revenue) * 100, 1) AS rev_growth_5yr_pct,
                ROUND(SUM(a.net_income) / 1000.0, 1) AS total_net_income_5yr_B
            FROM financials a
            JOIN companies c ON a.ticker = c.ticker
            GROUP BY a.ticker
        """,
    }
    results = {}
    for name, sql in queries.items():
        try:
            results[name] = pd.read_sql_query(sql, conn)
            print(f"  Query '{name}': {len(results[name])} rows")
        except Exception as e:
            print(f"  Query '{name}' failed: {e}")
            results[name] = pd.DataFrame()
    return results


# REVENUE FORECAST CHART
def plot_revenue_forecast(lnc_df, unm_df, proj_df):
    """Three-case scenario forecast chart for LNC and UNM."""
    fig = plt.figure(figsize=(20, 15))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

    fig.text(0.5, 0.97,
             "FP&A ANALYTICS SUITE — INSURANCE REVENUE FORECAST & SCENARIO ANALYSIS",
             ha="center", fontsize=16, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.945,
             "Lincoln National (NYSE: LNC) & Unum Group (NYSE: UNM) | "
             "FY2020–2024 Actuals + FY2025–2029 Projections | Source: SEC EDGAR",
             ha="center", fontsize=11, color=DGRAY, style="italic")

    ticker_colors = {"LNC": LNC_COLOR, "UNM": UNM_COLOR}

    for col, (ticker, df_act) in enumerate([("LNC", lnc_df), ("UNM", unm_df)]):
        color = ticker_colors[ticker]
        dn = proj_df[(proj_df["ticker"]==ticker) & (proj_df["scenario"]=="Downside")].sort_values("year")
        up = proj_df[(proj_df["ticker"]==ticker) & (proj_df["scenario"]=="Upside")].sort_values("year")

        # Revenue forecast
        ax = fig.add_subplot(gs[0, col*2:(col+1)*2])
        ax.bar(df_act["year"], df_act["revenue_B"],
               color=color, alpha=0.75, width=0.6, label="Actual")
        for sc_name in scenarios:
            p = proj_df[(proj_df["ticker"]==ticker) &
                        (proj_df["scenario"]==sc_name)].sort_values("year")
            ax.plot(p["year"], p["revenue_B"],
                    color=sc_colors[sc_name], ls=sc_ls[sc_name],
                    lw=2.5 if sc_name=="Base" else 2.0,
                    marker="o", ms=5, label=sc_name)
        ax.fill_between(proj_years,
                        dn["revenue_B"].values, up["revenue_B"].values,
                        alpha=0.08, color=color)
        ax.axvline(2024.5, color=DGRAY, ls="--", lw=1, alpha=0.5)
        ax.set_title(f"{COMPANIES_MAP[ticker]} — Revenue ($B)\nSource: SEC EDGAR 10-K",
                     fontweight="bold", color=NAVY, fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}B"))
        ax.legend(fontsize=8); ax.set_facecolor(LGRAY)

        # Operating margin
        ax2 = fig.add_subplot(gs[1, col*2:(col+1)*2])
        ax2.plot(df_act["year"], df_act["op_margin_pct"],
                 color=color, lw=2.5, marker="s", ms=7, label="Historical")
        for sc_name in scenarios:
            p = proj_df[(proj_df["ticker"]==ticker) &
                        (proj_df["scenario"]==sc_name)].sort_values("year")
            ax2.plot(p["year"], p["op_margin_pct"],
                     color=sc_colors[sc_name], ls=sc_ls[sc_name],
                     lw=2.5 if sc_name=="Base" else 2.0,
                     marker="o", ms=5, label=sc_name)
        ax2.fill_between(proj_years,
                         dn["op_margin_pct"].values, up["op_margin_pct"].values,
                         alpha=0.12, color=color)
        ax2.axvline(2024.5, color=DGRAY, ls="--", lw=1, alpha=0.5)
        ax2.set_title(f"{COMPANIES_MAP[ticker]} — Operating Margin %",
                      fontweight="bold", color=NAVY, fontsize=11)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax2.legend(fontsize=8); ax2.set_facecolor(LGRAY)

    # Side-by-side comparison + summary table
    ax3 = fig.add_subplot(gs[2, :2])
    x, w = np.arange(len(proj_years)), 0.35
    lnc_b = proj_df[(proj_df["ticker"]=="LNC") & (proj_df["scenario"]=="Base")].sort_values("year")
    unm_b = proj_df[(proj_df["ticker"]=="UNM") & (proj_df["scenario"]=="Base")].sort_values("year")
    ax3.bar(x - w/2, lnc_b["revenue_B"], w, color=LNC_COLOR, alpha=0.85, label="LNC Base")
    ax3.bar(x + w/2, unm_b["revenue_B"], w, color=UNM_COLOR, alpha=0.85, label="UNM Base")
    ax3.set_xticks(x); ax3.set_xticklabels(proj_years)
    ax3.set_title("Base Case Revenue Comparison ($B) 2025–2029",
                  fontweight="bold", color=NAVY, fontsize=11)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}B"))
    ax3.legend(fontsize=9); ax3.set_facecolor(LGRAY)

    # Summary table
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.axis("off")
    tbl = [["Year", "LNC Rev", "LNC Mgn", "UNM Rev", "UNM Mgn"]]
    for la, ua in zip(lnc_df.itertuples(), unm_df.itertuples()):
        tbl.append([str(la.year),
                    f"${la.revenue_B}B", f"{la.op_margin_pct}%",
                    f"${ua.revenue_B}B", f"{ua.op_margin_pct}%"])
    tbl.append(["---"] * 5)
    for lb, ub in zip(lnc_b.itertuples(), unm_b.itertuples()):
        tbl.append([f"{lb.year}*",
                    f"${lb.revenue_B}B", f"{lb.op_margin_pct}%",
                    f"${ub.revenue_B}B", f"{ub.op_margin_pct}%"])
    t = ax4.table(cellText=tbl[1:], colLabels=tbl[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.35)
    for j in range(5):
        t[0, j].set_facecolor(NAVY)
        t[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(tbl)):
        bg = LGRAY if i % 2 == 0 else "white"
        for j in range(5): t[i, j].set_facecolor(bg)
    for i in range(7, len(tbl)):
        for j in range(5): t[i, j].set_text_props(style="italic", color=DGRAY)
    ax4.set_title("5-Year Summary | *Base Case Projection",
                  fontweight="bold", color=NAVY, loc="left", fontsize=10)

    fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=7.5,
             color=DGRAY, style="italic")

    out = OUTPUT_DIR / "insurance_revenue_forecast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")



# EXPENSE DASHBOARD
def plot_expense_dashboard(query_results, lnc_df, unm_df):
    """9-panel expense reporting dashboard from SQL query results."""
    yoy  = query_results["annual_yoy"]
    rm   = query_results["rolling_margin"]
    expb = query_results["expense_breakdown"]
    qtr  = query_results["quarterly_2024"]
    kpi  = query_results["kpi_summary"]

    lnc_y = yoy[yoy["ticker"] == "LNC"]
    unm_y = yoy[yoy["ticker"] == "UNM"]
    lnc_r = rm[rm["ticker"]   == "LNC"]
    unm_r = rm[rm["ticker"]   == "UNM"]

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "FP&A ANALYTICS SUITE — INSURANCE EXPENSE REPORTING DASHBOARD\n"
        "Lincoln National (LNC) & Unum Group (UNM) | FY2020–2024 | Source: SEC EDGAR 10-K Filings",
        fontsize=14, fontweight="bold", color=NAVY, y=0.98)

    # Revenue trend
    ax = axes[0, 0]
    ax.plot(lnc_y["fiscal_year"], lnc_y["revenue"]/1000,
            color=LNC_COLOR, lw=2.5, marker="o", ms=7, label="LNC")
    ax.plot(unm_y["fiscal_year"], unm_y["revenue"]/1000,
            color=UNM_COLOR, lw=2.5, marker="s", ms=7, label="UNM")
    ax.set_title("Total Revenue ($B)", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}B"))
    ax.legend(fontsize=9)

    # Operating margin
    ax = axes[0, 1]
    ax.plot(lnc_y["fiscal_year"], lnc_y["op_margin_pct"],
            color=LNC_COLOR, lw=2.5, marker="o", ms=7, label="LNC")
    ax.plot(unm_y["fiscal_year"], unm_y["op_margin_pct"],
            color=UNM_COLOR, lw=2.5, marker="s", ms=7, label="UNM")
    ax.axhline(0, color=DGRAY, lw=1, ls="--")
    ax.set_title("Operating Margin %", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(fontsize=9)

    # Net income
    ax = axes[0, 2]
    x = np.arange(5); w = 0.35
    ax.bar(x - w/2, lnc_y["net_income"]/1000, w,
           color=LNC_COLOR, alpha=0.85, label="LNC")
    ax.bar(x + w/2, unm_y["net_income"]/1000, w,
           color=UNM_COLOR, alpha=0.85, label="UNM")
    ax.set_xticks(x); ax.set_xticklabels(lnc_y["fiscal_year"])
    ax.axhline(0, color=DGRAY, lw=1, ls="--")
    ax.set_title("Net Income ($B)", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}B"))
    ax.legend(fontsize=9)

    # YoY revenue growth
    ax = axes[1, 0]
    lg = lnc_y.dropna(subset=["yoy_rev_growth"])
    ug = unm_y.dropna(subset=["yoy_rev_growth"])
    ax.plot(lg["fiscal_year"], lg["yoy_rev_growth"],
            color=LNC_COLOR, lw=2.5, marker="o", ms=7, label="LNC")
    ax.plot(ug["fiscal_year"], ug["yoy_rev_growth"],
            color=UNM_COLOR, lw=2.5, marker="s", ms=7, label="UNM")
    ax.axhline(0, color=DGRAY, lw=1, ls="--")
    ax.set_title("YoY Revenue Growth %", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(fontsize=9)

    # LNC revenue vs expense growth
    ax = axes[1, 1]
    ax.plot(lg["fiscal_year"], lg["yoy_rev_growth"],
            color=GREEN, lw=2.5, marker="o", ms=7, label="Revenue Growth %")
    ax.plot(lg["fiscal_year"], lg["yoy_exp_growth"],
            color=RED,   lw=2.5, marker="s", ms=7, label="Expense Growth %")
    ax.axhline(0, color=DGRAY, lw=1, ls="--")
    ax.set_title("LNC — Revenue vs Expense Growth %", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(fontsize=9)

    # UNM expense stackplot
    ax = axes[1, 2]
    ax.stackplot(
        expb["fiscal_year"],
        expb["benefits"]    / 1000,
        expb["commissions"] / 1000,
        expb["other_opex"]  / 1000,
        labels=["Policy Benefits", "Commissions", "Other OpEx"],
        colors=[UNM_COLOR, AMBER, BLUE], alpha=0.85)
    ax.set_title("UNM — Expense Breakdown ($B)", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}B"))
    ax.legend(fontsize=8, loc="upper left")

    # FY2024 quarterly revenue
    ax = axes[2, 0]
    lq = qtr[qtr["ticker"] == "LNC"]
    uq = qtr[qtr["ticker"] == "UNM"]
    x = np.arange(4); w = 0.35
    ax.bar(x - w/2, lq["revenue"]/1000, w, color=LNC_COLOR, alpha=0.85, label="LNC")
    ax.bar(x + w/2, uq["revenue"]/1000, w, color=UNM_COLOR, alpha=0.85, label="UNM")
    ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_title("FY2024 Quarterly Revenue ($B)", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}B"))
    ax.legend(fontsize=9)

    # Rolling 3-year average margin
    ax = axes[2, 1]
    for ticker, color, df_r in [("LNC", LNC_COLOR, lnc_r), ("UNM", UNM_COLOR, unm_r)]:
        ax.plot(df_r["fiscal_year"], df_r["op_margin_pct"],
                color=color, lw=2, ls="--", alpha=0.6, marker="o", ms=5,
                label=f"{ticker} Actual")
        ax.plot(df_r["fiscal_year"], df_r["rolling_3yr_margin"],
                color=color, lw=2.5, marker="s", ms=7,
                label=f"{ticker} 3-Yr Avg")
    ax.axhline(0, color=DGRAY, lw=1, ls="--")
    ax.set_title("3-Year Rolling Avg Margin %", fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(fontsize=8)

    # KPI table
    ax = axes[2, 2]
    ax.axis("off")
    tbl_d = [
        [r["name"], f"${r['total_rev_5yr_B']}B", f"{r['avg_margin_5yr']}%",
         f"{r['rev_growth_5yr_pct']}%", f"${r['total_net_income_5yr_B']}B"]
        for _, r in kpi.iterrows()
    ]
    t = ax.table(
        cellText=tbl_d,
        colLabels=["Company", "5-Yr Rev", "Avg Margin", "5-Yr Growth", "5-Yr Net Inc"],
        loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.1, 2.8)
    for j in range(5):
        t[0, j].set_facecolor(NAVY)
        t[0, j].set_text_props(color="white", fontweight="bold")
    t[1, 0].set_facecolor(LGRAY)
    for j in range(1, 5):
        t[1, j].set_facecolor(LGRAY)
        t[2, j].set_facecolor("#D6E4F0")
    ax.set_title("5-Year KPI Summary", fontweight="bold", color=NAVY, pad=8)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor(LGRAY)

    fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=7.5,
             color=DGRAY, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    out = OUTPUT_DIR / "insurance_expense_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


# CSV & SQL EXPORTS
def export_csvs(actuals_df, proj_df):
    """Export all CSV files for Tableau and Power BI."""
    SOURCE = (
        "Annual figures: SEC EDGAR 10-K Filings (audited) | "
        "Quarterly: estimated from annual totals | "
        "Projections: modeled scenarios only — not investment advice"
    )
    actuals_df = actuals_df.copy()
    proj_df    = proj_df.copy()
    actuals_df["source"] = SOURCE
    proj_df["source"]    = SOURCE

    master = pd.concat([actuals_df, proj_df], ignore_index=True)
    master["data_source_note"] = SOURCE

    files = {
        "insurance_master_dataset.csv":    master,
        "insurance_actuals.csv":           actuals_df,
        "insurance_projections.csv":       proj_df,
    }
    for fname, df in files.items():
        path = OUTPUT_DIR / fname
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    # Expense breakdown
    exp_rows = []
    for ticker, df_e, cats in [
        ("LNC", lnc, [("Policy Benefits","benefits"),
                      ("Commissions","commissions"),
                      ("Other OpEx","other_opex")]),
        ("UNM", unm, [("Policy Benefits","benefits"),
                      ("Commissions","commissions"),
                      ("Other OpEx","other_opex")]),
    ]:
        for cat, col in cats:
            for _, row in df_e.iterrows():
                v = row[col]
                if v > 0:
                    exp_rows.append({
                        "ticker":           ticker,
                        "company":          row["company"],
                        "fiscal_year":      int(row["year"]),
                        "expense_category": cat,
                        "amount_M":         v,
                        "amount_B":         round(v / 1000, 2),
                        "pct_of_opex":      round(v / row["op_expenses"] * 100, 1),
                        "source":           SOURCE,
                    })
    path = OUTPUT_DIR / "insurance_expense_breakdown.csv"
    pd.DataFrame(exp_rows).to_csv(path, index=False)
    print(f"  Saved: {path}")


def export_sql():
    """Write all SQL queries to a standalone .sql file."""
    sql = """-- ================================================================
-- FP&A Analytics Suite — Insurance Company SQL Queries
-- Companies: Lincoln National (NYSE: LNC) & Unum Group (NYSE: UNM)
-- Source   : SEC EDGAR 10-K Annual Filings, FY2020-FY2024
-- Author   : Adedeji Adeleye, CA | Stonecroft Business and Financial Solutions
-- Note     : Uses SQLite window functions (LAG, AVG OVER, SUM OVER)
-- ================================================================

-- Q1: Annual summary with YoY growth using LAG window function
SELECT c.name, a.ticker, a.fiscal_year,
    a.revenue, a.op_expenses,
    ROUND((a.revenue - a.op_expenses) / a.revenue * 100, 1) AS op_margin_pct,
    a.op_income, a.net_income,
    ROUND((a.revenue - LAG(a.revenue) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year))
        / LAG(a.revenue) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year) * 100, 1) AS yoy_rev_growth,
    ROUND((a.op_expenses - LAG(a.op_expenses) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year))
        / LAG(a.op_expenses) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year) * 100, 1) AS yoy_exp_growth
FROM financials a
JOIN companies c ON a.ticker = c.ticker
ORDER BY a.ticker, a.fiscal_year;

-- Q2: Rolling 3-year average margin + cumulative revenue
SELECT c.name, a.ticker, a.fiscal_year,
    ROUND((a.revenue - a.op_expenses) / a.revenue * 100, 1) AS op_margin_pct,
    ROUND(AVG((a.revenue - a.op_expenses) / a.revenue * 100) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1) AS rolling_3yr_margin,
    ROUND(SUM(a.revenue) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year
        ROWS UNBOUNDED PRECEDING) / 1000.0, 1) AS cumulative_rev_B
FROM financials a
JOIN companies c ON a.ticker = c.ticker
ORDER BY a.ticker, a.fiscal_year;

-- Q3: UNM expense breakdown as % of total OpEx
SELECT fiscal_year,
    benefits,    ROUND(benefits    / op_expenses * 100, 1) AS benefits_pct,
    commissions, ROUND(commissions / op_expenses * 100, 1) AS comm_pct,
    other_opex,  ROUND(other_opex  / op_expenses * 100, 1) AS other_pct,
    op_expenses AS total_opex
FROM financials
WHERE ticker = 'UNM'
ORDER BY fiscal_year;

-- Q4: FY2024 quarterly breakdown with % of annual revenue
SELECT c.name, q.ticker, q.fiscal_year, q.quarter,
    q.revenue, q.op_expenses, q.op_income,
    ROUND(q.op_income / q.revenue * 100, 1) AS margin_pct,
    ROUND(q.revenue / SUM(q.revenue) OVER (
        PARTITION BY q.ticker, q.fiscal_year) * 100, 1) AS pct_of_annual
FROM quarterly q
JOIN companies c ON q.ticker = c.ticker
WHERE q.fiscal_year = 2024
ORDER BY q.ticker, q.quarter;

-- Q5: 5-year KPI summary by company
SELECT c.name, a.ticker,
    ROUND(SUM(a.revenue)    / 1000.0, 1) AS total_rev_5yr_B,
    ROUND(AVG((a.revenue - a.op_expenses) / a.revenue * 100), 1) AS avg_margin_5yr,
    ROUND((MAX(a.revenue) - MIN(a.revenue)) / MIN(a.revenue) * 100, 1) AS rev_growth_5yr_pct,
    ROUND(SUM(a.net_income) / 1000.0, 1) AS total_net_income_5yr_B
FROM financials a
JOIN companies c ON a.ticker = c.ticker
GROUP BY a.ticker;
"""
    path = OUTPUT_DIR / "insurance_queries.sql"
    with open(path, "w") as f:
        f.write(sql)
    print(f"  Saved: {path}")


# MAIN


def main():
    print("=" * 60)
    print("  FP&A Analytics Suite — Insurance Models")
    print("  Lincoln National (LNC) & Unum Group (UNM)")
    print("  Author: Adedeji Adeleye, CA | dejiadeleye.com")
    print("=" * 60)

    print("\n[1/5] Building database and running SQL queries...")
    conn    = build_database(lnc, unm)
    results = run_queries(conn)
    conn.close()

    print("\n[2/5] Building scenario projections...")
    print(f"  {len(proj)} projection rows across 2 companies x 3 scenarios x 5 years")

    print("\n[3/5] Plotting revenue forecast chart...")
    plot_revenue_forecast(lnc, unm, proj)

    print("\n[4/5] Plotting expense dashboard...")
    plot_expense_dashboard(results, lnc, unm)

    print("\n[5/5] Exporting CSV and SQL files...")
    export_csvs(actuals, proj)
    export_sql()

    print("\n" + "=" * 60)
    print(f"  Done. All files saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
