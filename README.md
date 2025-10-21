<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NinjaTrader Portfolio Analytics Dashboard</title>
<style>
* {margin:0;padding:0;box-sizing:border-box;}
body {
    font-family:'Bahnschrift Light','Bahnschrift',sans-serif;
    font-weight:300;
    line-height:1.8;
    color:#333;
    background-color:#f5f5f5;
    padding:40px 20px;
}
.container {
    max-width:900px;
    margin:0 auto;
    background-color:white;
    padding:50px;
    box-shadow:0 0 20px rgba(0,0,0,0.1);
}
h1 {
    font-size:2.5em;
    color:#0f172a;
    margin-bottom:25px;
    font-weight:400;
}
h2 {
    font-size:2em;
    color:#0f172a;
    margin-top:40px;
    margin-bottom:20px;
    padding-bottom:10px;
    border-bottom:2px solid #0f172a;
    font-weight:400;
}
h3 {
    font-size:1.5em;
    color:#1e293b;
    margin-top:30px;
    margin-bottom:15px;
    font-weight:400;
}
p {
    margin-bottom:20px;
    font-size:1.1em;
}
ul {
    margin-left:30px;
    margin-bottom:20px;
}
li {
    margin-bottom:10px;
    font-size:1.05em;
}
a {
    color:#dc2626;
    text-decoration:none;
}
a:hover {
    text-decoration:underline;
}
hr {
    border:none;
    border-top:1px solid #ddd;
    margin:40px 0;
}
.disclaimer {
    background-color:#fff3cd;
    border-left:4px solid:#ffc107;
    padding:20px;
    margin:30px 0;
}
.support {
    background-color:#e8f4f8;
    border-left:4px solid:#0dcaf0;
    padding:20px;
    margin:30px 0;
}
.license {
    margin-top:30px;
}
.license img {
    margin-top:10px;
}
.footer {
    text-align:center;
    color:#666;
    font-style:italic;
    margin-top:40px;
    padding-top:30px;
    border-top:1px solid #ddd;
}
</style>
</head>
<body>
<div class="container">
    <h1>NinjaTrader Portfolio Analytics Dashboard</h1>
    
    <p>Algorithmic traders running multiple strategies need a single view to understand how strategies interact in one account: which strategies drive returns, which increase risk, and how allocations impact portfolio outcomes. NinjaTrader's native Strategy Analyser does not provide this functionality.</p>
    
    <h2>Solution:</h2>
    <p>A Streamlit dashboard for NinjaTrader users to:</p>
    <ul>
        <li>Upload period-summary CSVs</li>
        <li>Analyze individual strategies and combined portfolio</li>
        <li>Test custom allocation weights in real-time</li>
        <li>Visualize equity curves, drawdowns, correlations, and strategy contributions</li>
    </ul>
    
    <h2>Key Features</h2>
    <ul>
        <li><strong>Strategy Analytics:</strong> Profit, win rate, expectancy, MAE/MFE, drawdowns</li>
        <li><strong>Portfolio View:</strong> Combined equity, strategy contributions, rolling metrics</li>
        <li><strong>Risk & Diversification:</strong> Max drawdown, Sharpe/Sortino, recovery factor, correlation heatmap, diversification scoring</li>
        <li><strong>Comparison Tools:</strong> Side-by-side strategy and benchmark comparisons</li>
        <li><strong>Exports:</strong> Download charts and CSV metrics</li>
    </ul>
    
    <h3>Usage:</h3>
    <ol>
        <li>Export NinjaTrader CSV → Upload to dashboard</li>
        <li>Optional: Set custom allocation weights</li>
        <li>Explore dashboards and export results</li>
    </ol>
    
    <h3>Metrics:</h3>
    <ul>
        <li><strong>Performance:</strong> Net profit, Profit factor, Win rate</li>
        <li><strong>Risk:</strong> Max drawdown, Sharpe, Sortino, Recovery factor</li>
        <li><strong>Portfolio:</strong> Correlation, diversification score</li>
    </ul>
    
    <div class="disclaimer">
        <h2 style="margin-top:0;border:none;">Disclaimer</h2>
        <p style="margin-bottom:0;">This tool is for educational and analytical purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.</p>
    </div>
    
    <div class="support">
        <h2 style="margin-top:0;border:none;">Support</h2>
        <p style="margin-bottom:0;">For questions or issues: <a href="https://www.linkedin.com/in/meronmkifle/" target="_blank">https://www.linkedin.com/in/meronmkifle/</a></p>
    </div>
    
    <div class="license">
        <h2>License</h2>
        <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
            <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
        </a>
        <p>This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.</p>
    </div>
    
    <hr>
    
    <div class="footer">
        Built for NinjaTrader traders | Powered by Streamlit
    </div>
</div>
</body>
</html>
