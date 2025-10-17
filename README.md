# NinjaTrader Portfolio Analytics Dashboard

A fairly comprehensive trading analytics dashboard built with Streamlit for NinjaTrader users who trade multiple algorithmic strategies. Analyze individual strategy performance and see how strategies work together in a combined trading account.

## What It Does

TradeLens Pro helps algorithmic traders answer critical questions:

- How does each strategy perform individually?
- What happens when I run multiple strategies in one account?
- Are my strategies correlated or diversified?
- Does combining strategies reduce risk?
- How should I allocate capital between strategies?

The dashboard processes NinjaTrader period summary exports and provides portfolio-level analytics with professional visualizations.

## Key Features

### Individual Strategy Analysis
- Complete performance metrics (profit, win rate, Sharpe ratio, Sortino ratio)
- Risk analysis (max drawdown, recovery factor)
- Period-by-period return distribution
- Equity curve visualization

### Combined Account View
- **See all strategies running in ONE account**
- Combined equity curve showing total portfolio performance
- Strategy contribution breakdown (pie & bar charts)
- Period-by-period combined returns
- Win/loss streak analysis
- Rolling performance metrics

### Portfolio Analytics
- Strategy correlation heatmap
- Diversification benefit scoring (Excellent/Good/Fair/Limited)
- Combined vs individual metrics comparison
- Position sizing with custom allocation weights

### Comparison Tools
- Side-by-side strategy comparison
- Buy & Hold benchmark comparison (ES=F, NQ=F, SPY, etc.)
- Monthly equity curves
- Drawdown analysis across strategies


## Usage

### Exporting from NinjaTrader

1. Open **Control Center** in NinjaTrader
2. Navigate to the **Performance** tab
3. Set your desired time period
4. Right-click and select **Export**
5. Save as CSV file
6. Upload to TradeLens Pro

### Uploading Files

1. Click "Upload CSV/Excel Files" in the sidebar
2. Select one or more NinjaTrader export files
3. Name your strategies (optional)
4. Dashboard automatically processes and displays analytics

### Custom Allocation (Optional)

1. Check "Custom Weight" in Position Sizing section
2. Set allocation percentage for each strategy
3. Percentages must sum to 100%
4. All charts update in real-time

## Dashboard Tabs

### 1. Dashboard
- Portfolio overview with key metrics
- Strategy performance summary table
- Quick access to all metrics

### 2. Combined Account 
- **Main feature**: Shows how all strategies perform together
- Combined equity curve
- Strategy contribution breakdown
- Rolling performance analysis
- Diversification benefits summary

### 3. Individual Strategies
- Deep dive into each strategy
- Complete performance metrics
- Risk analysis and trade statistics

### 4. Portfolio Analysis
- Strategy correlation heatmap
- Diversification scoring
- Combined portfolio metrics

### 5. Equity Curves
- Monthly performance comparison
- All strategies + combined + benchmark
- Interactive visualizations

### 6. Drawdown Analysis
- Equity curve with drawdown overlay
- Maximum drawdown statistics
- Drawdown duration tracking

### 7. Period Analysis
- Period return distribution
- Box plots and histograms
- Statistical analysis
- Export functionality

## Understanding Key Metrics

### Performance Metrics
- **Net Profit**: Total profit after commissions
- **Profit Factor**: Gross profits / Gross losses (>1.5 is good)
- **Win Rate**: Percentage of profitable periods
- **Expectancy**: Average profit per period

### Risk Metrics
- **Max Drawdown**: Largest peak-to-trough decline in account value
- **Sharpe Ratio**: Risk-adjusted return (>1 is good, >2 is excellent)
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
- **Recovery Factor**: Net profit / Max drawdown

### Portfolio Metrics
- **Correlation**: How strategies move together (-1 to +1)
- **Diversification Benefit**: Quality rating based on correlation
  - Excellent: Average correlation <0.3
  - Good: Average correlation 0.3-0.5
  - Fair: Average correlation 0.5-0.7
  - Limited: Average correlation >0.7


## File Format

The dashboard expects NinjaTrader period summary CSV files with these columns:
- Period (date)
- Cum. net profit
- Net profit
- Gross profit / Gross loss
- % Win
- Avg. trade, Avg. winner, Avg. loser
- Max. drawdown, Cum. max. drawdown
- Commission
- Avg. MAE, Avg. MFE

The dashboard automatically detects and parses NinjaTrader format.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


## Disclaimer

This tool is for educational and analytical purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## Support

For questions or issues:
- Any inquiries: https://www.linkedin.com/in/meronmkifle/
- Documentation: See this README

## Version

Current Version: 2.0

Last Updated: October 2025

---

Built for NinjaTrader traders | Powered by Streamlit
