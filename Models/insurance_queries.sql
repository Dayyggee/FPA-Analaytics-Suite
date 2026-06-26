-- ================================================================
-- FP&A Analytics Suite — Insurance Company SQL Queries
-- Companies: Lincoln National (NYSE: LNC) & Unum Group (NYSE: UNM)
-- Source   : SEC EDGAR 10-K Annual Filings, FY2020-FY2024
-- Author   : Djay Smith, CA | Stonecroft Business and Financial Solutions
-- ================================================================

-- Q1: Annual summary with YoY growth (LAG window function)

SELECT c.name, a.ticker, a.fiscal_year,
    a.revenue, a.op_expenses,
    ROUND((a.revenue-a.op_expenses)/a.revenue*100,1)   AS op_margin_pct,
    a.op_income, a.net_income,
    ROUND((a.revenue - LAG(a.revenue) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year))
        / LAG(a.revenue) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year)*100,1) AS yoy_rev_growth,
    ROUND((a.op_expenses - LAG(a.op_expenses) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year))
        / LAG(a.op_expenses) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year)*100,1) AS yoy_exp_growth
FROM financials a JOIN companies c ON a.ticker=c.ticker
ORDER BY a.ticker, a.fiscal_year


-- Q2: Rolling 3-year average margin + cumulative revenue

SELECT c.name, a.ticker, a.fiscal_year,
    ROUND((a.revenue-a.op_expenses)/a.revenue*100,1) AS op_margin_pct,
    ROUND(AVG((a.revenue-a.op_expenses)/a.revenue*100) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),1) AS rolling_3yr_margin,
    ROUND(SUM(a.revenue) OVER (
        PARTITION BY a.ticker ORDER BY a.fiscal_year
        ROWS UNBOUNDED PRECEDING)/1000.0,1)          AS cumulative_rev_B
FROM financials a JOIN companies c ON a.ticker=c.ticker
ORDER BY a.ticker, a.fiscal_year


-- Q3: UNM expense breakdown as % of total OpEx

SELECT fiscal_year,
    benefits, ROUND(benefits/op_expenses*100,1)      AS benefits_pct,
    commissions, ROUND(commissions/op_expenses*100,1) AS comm_pct,
    other_opex, ROUND(other_opex/op_expenses*100,1)  AS other_pct,
    op_expenses AS total_opex
FROM financials WHERE ticker='UNM' ORDER BY fiscal_year


-- Q4: FY2024 quarterly breakdown with % of annual revenue

SELECT c.name, q.ticker, q.fiscal_year, q.quarter,
    q.revenue, q.op_expenses, q.op_income,
    ROUND(q.op_income/q.revenue*100,1) AS margin_pct,
    ROUND(q.revenue/SUM(q.revenue) OVER (
        PARTITION BY q.ticker,q.fiscal_year)*100,1)  AS pct_of_annual
FROM quarterly q JOIN companies c ON q.ticker=c.ticker
WHERE q.fiscal_year=2024 ORDER BY q.ticker, q.quarter


-- Q5: 5-year KPI summary by company

SELECT c.name, a.ticker,
    ROUND(SUM(a.revenue)/1000.0,1)                   AS total_rev_5yr_B,
    ROUND(AVG((a.revenue-a.op_expenses)/a.revenue*100),1) AS avg_margin_5yr,
    ROUND((MAX(a.revenue)-MIN(a.revenue))/MIN(a.revenue)*100,1) AS rev_growth_5yr_pct,
    ROUND(SUM(a.net_income)/1000.0,1)                AS total_net_income_5yr_B
FROM financials a JOIN companies c ON a.ticker=c.ticker
GROUP BY a.ticker
