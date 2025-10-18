# NinjaTrader Portfolio Analytics Dashboard

Algorithmic traders running multiple strategies need a single view to understand how strategies interact in one account: which strategies drive returns, which increase risk, and how allocations impact portfolio outcomes. NinjaTrader’s native Strategy Analyser does not provide this functionality.

## Solution:

A Streamlit dashboard for NinjaTrader users to:

- Upload period-summary CSVs
- Analyze individual strategies and combined portfolio
- Test custom allocation weights in real-time
- Visualize equity curves, drawdowns, correlations, and strategy contributions

## Key Features

- Strategy Analytics: Profit, win rate, expectancy, MAE/MFE, drawdowns
- Portfolio View: Combined equity, strategy contributions, rolling metrics
- Risk & Diversification: Max drawdown, Sharpe/Sortino, recovery factor, correlation heatmap, diversification scoring
- Comparison Tools: Side-by-side strategy and benchmark comparisons
- Exports: Download charts and CSV metrics

### Usage:
1. Export NinjaTrader CSV → Upload to dashboard
2. Optional: Set custom allocation weights
3. Explore dashboards and export results

### Metrics:
- Performance: Net profit, Profit factor, Win rate
- Risk: Max drawdown, Sharpe, Sortino, Recovery factor
- Portfolio: Correlation, diversification score


## Disclaimer

This tool is for educational and analytical purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## Support

For questions or issues: https://www.linkedin.com/in/meronmkifle/
  
## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.


---

Built for NinjaTrader traders | Powered by Streamlit
