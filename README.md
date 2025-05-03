# Cont & Kukanov Smart Order Router Back-testing

This repository contains an implementation of the Cont & Kukanov Smart Order Router model for back-testing against historical market data, as described in their paper "Optimal Order Placement in Limit Order Markets".

## Code Structure

The implementation follows these main components:

1. **Data Preprocessing**: Parses the L1 market data, keeping only the first message per venue per timestamp and extracting relevant fields (ask price and size).

2. **Allocator Implementation**: Implements the allocation algorithm exactly as described in the pseudocode, which splits orders across multiple venues based on the cost model.

3. **Back-testing Framework**: Simulates order execution over the historical data using the allocator's decisions.

4. **Parameter Search**: Conducts a grid search over lambda_over, lambda_under, and theta_queue to find the optimal parameter set.

5. **Baseline Strategies**: Implements three baseline strategies (best-ask, TWAP, and VWAP) for comparison.

6. **Results Visualization**: Creates a cumulative cost plot showing the performance of each strategy.

## Parameter Search Approach

The search space is defined as a grid across these parameters:

- **lambda_over**: [0.01, 0.05, 0.1, 0.2, 0.5]  
  Penalty for exceeding the target quantity. Higher values discourage over-buying.

- **lambda_under**: [0.01, 0.05, 0.1, 0.2, 0.5]  
  Penalty for not reaching the target quantity. Higher values prioritize execution completion.

- **theta_queue**: [0.001, 0.005, 0.01, 0.05, 0.1]  
  Queue-risk penalty that linearly scales with total mis-execution. Controls trade-off between price and execution certainty.

These ranges were chosen to cover reasonable values based on the paper's description, spanning from minimal penalties to significant ones. The search aims to find the combination that minimizes the average execution price.
