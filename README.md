# ANTICOR Algorithm Replication and Analysis

This repository contains the Python implementation, backtesting engine, and critical analysis of the ANTICOR portfolio selection algorithm (Borodin et al., 2004).



READ THE NOTES BEFORE TOUCHING THE CODE:
[Important Notes for Collaborators](./NOTES.md)

## Project Objectives

1.  **Algorithmic Replication:** Develop a robust Python backtester for the core $ANTICOR_w$ algorithm and its "smoothed" variations, $ANTI^1$ and $ANTI^2$.
2.  **Empirical Validation:** Validate our implementation by replicating the key results from the original paper (Table 1) on our chosen dataset.
3.  **Critical Assessment:** Evaluate the algorithm's sensitivity to transaction costs and its robustness through rigorous out-of-sample (OOS) testing.

## Methodology Overview

### 1. Data Pipeline

* **Source:** Center for Research in Security Prices (CRSP) via WRDS.
* **Replication Universe:** 25 largest S&P 500 market-cap stocks.
* **In-Sample Period:** 1276 trading days ending January 30, 2025 (start date ~August 20, 2019).
* **Cleaning & Transformation:**
    1.  Extract `PERMNO`, `DATE`, `RET`, `DLRET`, `PRC`.
    2.  Calculate **Total Effective Return** to correctly handle delisting events: $(1 + \text{RET}) \times (1 + \text{DLRET}) - 1$.
    3.  Construct the final master matrix: Dates (index) x Stocks (columns), containing daily price relatives. Handle missing data via forward-filling.

### 2. Backtest Implementation

* **Initialization:** A $2w$ "warm-up" period is used to prime the algorithm before trading begins.
* **Core Logic ($ANTICOR_w$):**
    1.  On day $t$, construct the $LX_1$ (past) and $LX_2$ (recent) log-relative price matrices.
    2.  Compute statistical measures: $\mu_1, \mu_2, \sigma_1, \sigma_2$.
    3.  Compute the core $m \times m$ cross-correlation matrix, $M_{cor}$.
    4.  Calculate the $claim_{i \to j}$ matrix based on gating conditions:
        * Performance: $\mu_2(i) > \mu_2(j)$
        * Correlation: $M_{cor}(i, j) > 0$
    5.  Calculate $b_{t+1}$ by executing wealth transfers based on claim scores.
* **Smoothed Strategies (Ensemble):**
    * **Parameter:** $W=30$.
    * **$ANTI^1$:** Arithmetic mean of the $W-1$ "expert" portfolios ($w \in [2, W]$).
    * **$ANTI^2$:** A recursive application of ANTICOR, trading on the return streams of the $W-1$ experts.

### 3. Analysis Framework

* **Transaction Costs:** Implement the proportional cost model (using $\gamma$) to conduct a sensitivity analysis and find the strategy's break-even commission rate.
* **Out-of-Sample (OOS) Test:** Apply the finalized, unmodified algorithm to a completely different, non-overlapping historical period (e.g., 2010-2018) to test for robustness and mitigate data-snooping bias.
