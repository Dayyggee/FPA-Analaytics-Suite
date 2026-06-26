-- ================================================================
-- FP&A Analytics Suite — SQL Query Library
-- Author  : Djay Smith, CA | Stonecroft Business and Financial Solutions
-- Source  : HCA Healthcare & Humana Inc. 10-K Filings, SEC EDGAR
-- Note    : Uses SQLite window functions (LAG, AVG OVER, SUM OVER)
-- ================================================================


-- Q1: Annual summary with YoY growth using LAG window function
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


-- Q2: 5-year KPI rollup with 3-year rolling average margin
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


-- Q3: HCA expense category breakdown as % of total OpEx
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


-- Q4: Quarterly breakdown for FY2024 with % of annual revenue
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


-- Q5: Margin trend with cumulative revenue and income (running totals)
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
