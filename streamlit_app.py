import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats

st.set_page_config(page_title="TradeLens Pro - NinjaTrader Analytics", layout="wide", page_icon="📊")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def parse_ninjatrader_file(file):
    """Parse NinjaTrader export files (CSV or Excel)"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Common NinjaTrader column mappings
        column_mapping = {
            'Entry Time': 'Entry time',
            'Entry time': 'Entry time',
            'Exit Time': 'Exit time',
            'Exit time': 'Exit time',
            'Instrument': 'Symbol',
            'Market pos.': 'Direction',
            'Market Position': 'Direction',
            'Quantity': 'Contracts',
            'Entry price': 'Entry Price',
            'Exit price': 'Exit Price',
            'Profit': 'Profit',
            'Cum. Profit': 'Cum. net profit',
            'Cumulative': 'Cum. net profit',
            'Commission': 'Commission'
        }
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def process_trading_data(df):
    """Process and clean trading data"""
    # Parse dates
    date_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    for date_col in date_cols:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Clean numeric columns
    numeric_cols = ['Profit', 'Cum. net profit', 'Entry Price', 'Exit Price', 'Commission']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('%', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract time features
    if 'Entry time' in df.columns:
        df['Month_Year'] = df['Entry time'].dt.strftime('%Y-%m')
        df['Date'] = df['Entry time'].dt.date
        df['Day_of_Week'] = df['Entry time'].dt.day_name()
        df['Hour'] = df['Entry time'].dt.hour
        df['Week'] = df['Entry time'].dt.isocalendar().week
        df['Year'] = df['Entry time'].dt.year
        df['Time_in_Trade'] = (df['Exit time'] - df['Entry time']).dt.total_seconds() / 60 if 'Exit time' in df.columns else None
    
    # Calculate additional metrics
    if 'Profit' in df.columns:
        df['Win'] = df['Profit'] > 0
        df['Loss'] = df['Profit'] < 0
        
    # Calculate R-multiples if possible
    if 'Profit' in df.columns and 'Entry Price' in df.columns:
        df['R_Multiple'] = df['Profit'] / (df['Entry Price'] * 0.01)  # Assuming 1% risk
    
    return df

def calculate_advanced_metrics(df):
    """Calculate comprehensive trading metrics"""
    metrics = {}
    
    if 'Profit' not in df.columns or len(df) == 0:
        return metrics
    
    # Basic metrics
    total_trades = len(df)
    winning_trades = len(df[df['Profit'] > 0])
    losing_trades = len(df[df['Profit'] < 0])
    
    metrics['Total Trades'] = total_trades
    metrics['Winning Trades'] = winning_trades
    metrics['Losing Trades'] = losing_trades
    metrics['Win Rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Profit metrics
    wins = df[df['Profit'] > 0]['Profit']
    losses = df[df['Profit'] < 0]['Profit']
    
    metrics['Avg Win'] = wins.mean() if len(wins) > 0 else 0
    metrics['Avg Loss'] = losses.mean() if len(losses) > 0 else 0
    metrics['Largest Win'] = wins.max() if len(wins) > 0 else 0
    metrics['Largest Loss'] = losses.min() if len(losses) > 0 else 0
    
    metrics['Total Profit'] = df['Profit'].sum()
    metrics['Total Wins'] = wins.sum() if len(wins) > 0 else 0
    metrics['Total Losses'] = abs(losses.sum()) if len(losses) > 0 else 0
    
    # Profit Factor
    metrics['Profit Factor'] = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
    
    # Expectancy
    metrics['Expectancy'] = df['Profit'].mean()
    metrics['Expectancy %'] = (metrics['Win Rate'] / 100 * metrics['Avg Win'] + 
                                (1 - metrics['Win Rate'] / 100) * metrics['Avg Loss'])
    
    # Risk-Reward Ratio
    metrics['Avg RR Ratio'] = abs(metrics['Avg Win'] / metrics['Avg Loss']) if metrics['Avg Loss'] != 0 else 0
    
    # Cumulative metrics
    if 'Cum. net profit' in df.columns:
        metrics['Net Profit'] = df['Cum. net profit'].iloc[-1]
        metrics['Peak Profit'] = df['Cum. net profit'].max()
        
        # Drawdown
        cumulative = df['Cum. net profit']
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        metrics['Max Drawdown'] = drawdown.min()
        metrics['Max Drawdown %'] = (drawdown / running_max * 100).min() if running_max.max() > 0 else 0
        
        # Recovery Factor
        metrics['Recovery Factor'] = abs(metrics['Net Profit'] / metrics['Max Drawdown']) if metrics['Max Drawdown'] != 0 else 0
    
    # Consistency metrics
    if 'Month_Year' in df.columns:
        monthly_profits = df.groupby('Month_Year')['Profit'].sum()
        metrics['Profitable Months'] = len(monthly_profits[monthly_profits > 0])
        metrics['Total Months'] = len(monthly_profits)
        metrics['Monthly Win Rate'] = (metrics['Profitable Months'] / metrics['Total Months'] * 100) if metrics['Total Months'] > 0 else 0
    
    # Statistical metrics
    metrics['Profit Std Dev'] = df['Profit'].std()
    metrics['Sharpe Ratio'] = (df['Profit'].mean() / df['Profit'].std() * np.sqrt(252)) if df['Profit'].std() > 0 else 0
    
    # Sortino Ratio
    downside_returns = df[df['Profit'] < 0]['Profit']
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
    metrics['Sortino Ratio'] = (df['Profit'].mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    # Kelly Criterion
    if metrics['Win Rate'] > 0 and metrics['Avg RR Ratio'] > 0:
        win_prob = metrics['Win Rate'] / 100
        metrics['Kelly %'] = (win_prob - ((1 - win_prob) / metrics['Avg RR Ratio'])) * 100
    else:
        metrics['Kelly %'] = 0
    
    # Average time in trade
    if 'Time_in_Trade' in df.columns:
        metrics['Avg Time in Trade (min)'] = df['Time_in_Trade'].mean()
    
    # Consecutive wins/losses
    df['Streak'] = (df['Profit'] > 0).astype(int)
    df['Streak_Group'] = (df['Streak'] != df['Streak'].shift()).cumsum()
    streaks = df.groupby(['Streak', 'Streak_Group']).size()
    
    winning_streaks = streaks[1] if 1 in streaks.index.get_level_values(0) else pd.Series([0])
    losing_streaks = streaks[0] if 0 in streaks.index.get_level_values(0) else pd.Series([0])
    
    metrics['Max Consecutive Wins'] = winning_streaks.max()
    metrics['Max Consecutive Losses'] = losing_streaks.max()
    
    return metrics

def calculate_portfolio_metrics(dataframes):
    """Calculate metrics for combined portfolio"""
    if not dataframes:
        return {}
    
    # Combine all trades chronologically
    all_trades = []
    for name, df in dataframes.items():
        if 'Entry time' in df.columns and 'Profit' in df.columns:
            temp_df = df[['Entry time', 'Profit']].copy()
            temp_df['Strategy'] = name
            all_trades.append(temp_df)
    
    if not all_trades:
        return {}
    
    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values('Entry time').reset_index(drop=True)
    combined['Cum_Profit'] = combined['Profit'].cumsum()
    
    # Calculate correlation between strategies
    correlations = {}
    strategy_names = list(dataframes.keys())
    for i, name1 in enumerate(strategy_names):
        for name2 in strategy_names[i+1:]:
            df1 = dataframes[name1]
            df2 = dataframes[name2]
            
            if 'Profit' in df1.columns and 'Profit' in df2.columns:
                corr = df1['Profit'].corr(df2['Profit'])
                correlations[f"{name1} vs {name2}"] = corr
    
    # Portfolio metrics
    portfolio_metrics = calculate_advanced_metrics(combined)
    portfolio_metrics['Strategy Correlations'] = correlations
    portfolio_metrics['Number of Strategies'] = len(dataframes)
    portfolio_metrics['Diversification Benefit'] = 'High' if all(abs(c) < 0.5 for c in correlations.values()) else 'Medium' if all(abs(c) < 0.7 for c in correlations.values()) else 'Low'
    
    return portfolio_metrics, combined

def monte_carlo_simulation(df, n_simulations=1000, n_trades=None):
    """Run Monte Carlo simulation on trade results"""
    if 'Profit' not in df.columns or len(df) == 0:
        return None
    
    if n_trades is None:
        n_trades = len(df)
    
    profits = df['Profit'].values
    simulations = []
    
    for _ in range(n_simulations):
        sim_trades = np.random.choice(profits, size=n_trades, replace=True)
        sim_cumulative = np.cumsum(sim_trades)
        simulations.append(sim_cumulative[-1])
    
    return np.array(simulations)

# ==================== STREAMLIT APP ====================

st.markdown('<h1 class="main-header">🎯 TradeLens Pro - NinjaTrader Analytics</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=TradeLens+Pro", use_container_width=True)
    st.markdown("---")
    
    st.header("📁 Upload Trading Data")
    st.markdown("*Supports NinjaTrader CSV/Excel exports*")
    
    uploaded_files = st.file_uploader(
        "Upload Files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload NinjaTrader performance reports"
    )
    
    st.markdown("---")
    
    if uploaded_files:
        st.subheader("Strategy Names")
        strategy_names = {}
        for i, file in enumerate(uploaded_files):
            default_name = file.name.replace('.csv', '').replace('.xlsx', '').replace('.xls', '')
            strategy_names[i] = st.text_input(
                f"Strategy {i+1}",
                value=default_name,
                key=f"name_{i}",
                label_visibility="collapsed"
            )
    
    st.markdown("---")
    
    # Analysis Options
    st.subheader("⚙️ Analysis Options")
    
    show_benchmark = st.checkbox("Compare with Benchmark", value=False)
    if show_benchmark:
        benchmark_ticker = st.text_input("Benchmark Ticker", value="ES=F")
        benchmark_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
    
    show_monte_carlo = st.checkbox("Monte Carlo Simulation", value=False)
    if show_monte_carlo:
        n_simulations = st.slider("Number of Simulations", 100, 5000, 1000, step=100)
    
    st.markdown("---")
    st.caption("Built for NinjaTrader | Version 1.0")

# Main Content
if not uploaded_files:
    st.info("👆 Upload your NinjaTrader trading files to begin analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Individual Analytics")
        st.markdown("- Comprehensive performance metrics")
        st.markdown("- Risk-adjusted returns")
        st.markdown("- Trade distribution analysis")
    
    with col2:
        st.markdown("### 🔄 Portfolio Analytics")
        st.markdown("- Combined strategy performance")
        st.markdown("- Strategy correlation analysis")
        st.markdown("- Diversification benefits")
    
    with col3:
        st.markdown("### 🎲 Advanced Features")
        st.markdown("- Monte Carlo simulations")
        st.markdown("- Benchmark comparisons")
        st.markdown("- Trade expectancy analysis")
    
    st.markdown("---")
    st.markdown("### 📋 NinjaTrader Export Instructions")
    st.markdown("""
    1. In NinjaTrader, go to **Control Center > Account Performance**
    2. Right-click on your account and select **Performance**
    3. Click **Export** and save as CSV or Excel
    4. Upload the file here for instant analytics
    """)

else:
    # Load and process all files
    dataframes = {}
    
    for i, file in enumerate(uploaded_files):
        df = parse_ninjatrader_file(file)
        if df is not None:
            strategy_name = strategy_names.get(i, f"Strategy {i+1}")
            df = process_trading_data(df)
            dataframes[strategy_name] = df
    
    if not dataframes:
        st.error("No valid data found in uploaded files. Please check your file format.")
        st.stop()
    
    # Create tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "🎯 Individual Strategies",
        "🔄 Portfolio Analysis",
        "📈 Performance Comparison",
        "📉 Drawdown Analysis",
        "📅 Trade Distribution",
        "🎲 Monte Carlo",
        "📄 Detailed Reports"
    ])
    
    # ==================== TAB 1: DASHBOARD ====================
    with tabs[0]:
        st.header("Performance Dashboard")
        
        # Calculate portfolio metrics
        portfolio_metrics, combined_trades = calculate_portfolio_metrics(dataframes)
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Net Profit",
                f"${portfolio_metrics.get('Net Profit', 0):,.2f}",
                delta=f"{portfolio_metrics.get('Win Rate', 0):.1f}% Win Rate"
            )
        
        with col2:
            st.metric(
                "Total Trades",
                f"{portfolio_metrics.get('Total Trades', 0):,}",
                delta=f"{portfolio_metrics.get('Number of Strategies', 0)} Strategies"
            )
        
        with col3:
            st.metric(
                "Profit Factor",
                f"{portfolio_metrics.get('Profit Factor', 0):.2f}",
                delta="Good" if portfolio_metrics.get('Profit Factor', 0) > 1.5 else "Fair"
            )
        
        with col4:
            st.metric(
                "Sharpe Ratio",
                f"{portfolio_metrics.get('Sharpe Ratio', 0):.2f}",
                delta="Excellent" if portfolio_metrics.get('Sharpe Ratio', 0) > 1 else "Fair"
            )
        
        with col5:
            st.metric(
                "Max Drawdown",
                f"${portfolio_metrics.get('Max Drawdown', 0):,.2f}",
                delta=f"{portfolio_metrics.get('Max Drawdown %', 0):.1f}%",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Equity Curve
        st.subheader("Combined Equity Curve")
        
        monthly_data = {}
        for name, df in dataframes.items():
            if 'Month_Year' in df.columns and 'Cum. net profit' in df.columns:
                monthly = df.groupby('Month_Year')['Cum. net profit'].last().sort_index()
                monthly_data[name] = monthly
        
        if monthly_data:
            combined_df = pd.DataFrame(monthly_data)
            combined_df = combined_df.ffill().fillna(0)
            combined_df['Portfolio'] = combined_df.sum(axis=1)
            
            # Add benchmark if selected
            if show_benchmark and benchmark_ticker:
                try:
                    start_date = combined_df.index.min() + '-01'
                    end_date = combined_df.index.max() + '-28'
                    
                    market_data = yf.download(benchmark_ticker, start=start_date, end=end_date, 
                                            interval="1mo", progress=False)
                    if not market_data.empty:
                        market_data['Month_Year'] = market_data.index.strftime('%Y-%m')
                        monthly_prices = market_data.groupby('Month_Year')['Close'].last()
                        
                        first_price = monthly_prices.iloc[0]
                        buy_hold_returns = ((monthly_prices / first_price) - 1) * benchmark_capital
                        combined_df['Benchmark'] = buy_hold_returns.reindex(combined_df.index).ffill().fillna(0)
                except:
                    pass
            
            # Plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            for col in combined_df.columns:
                if col == 'Portfolio':
                    ax.plot(combined_df.index, combined_df[col], label=col, 
                           linewidth=2.5, color='#2ecc71')
                elif col == 'Benchmark':
                    ax.plot(combined_df.index, combined_df[col], label=col, 
                           linewidth=2, linestyle='--', color='#e74c3c')
                else:
                    ax.plot(combined_df.index, combined_df[col], label=col, 
                           linewidth=1.5, alpha=0.7)
            
            ax.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Month-Year', fontsize=12)
            ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
            
            # Format x-axis
            x_ticks = range(0, len(combined_df.index), max(1, len(combined_df.index) // 12))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([combined_df.index[i] for i in x_ticks], rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Performance summary table
            st.subheader("Strategy Performance Summary")
            summary_data = []
            for name, df in dataframes.items():
                metrics = calculate_advanced_metrics(df)
                summary_data.append({
                    'Strategy': name,
                    'Net Profit': f"${metrics.get('Net Profit', 0):,.2f}",
                    'Trades': metrics.get('Total Trades', 0),
                    'Win Rate': f"{metrics.get('Win Rate', 0):.1f}%",
                    'Profit Factor': f"{metrics.get('Profit Factor', 0):.2f}",
                    'Sharpe': f"{metrics.get('Sharpe Ratio', 0):.2f}",
                    'Max DD': f"${metrics.get('Max Drawdown', 0):,.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # ==================== TAB 2: INDIVIDUAL STRATEGIES ====================
    with tabs[1]:
        st.header("Individual Strategy Analysis")
        
        selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()))
        df = dataframes[selected_strategy]
        metrics = calculate_advanced_metrics(df)
        
        # Metrics Grid
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### Trading Activity")
            st.metric("Total Trades", f"{metrics.get('Total Trades', 0):,}")
            st.metric("Winning Trades", f"{metrics.get('Winning Trades', 0):,}")
            st.metric("Losing Trades", f"{metrics.get('Losing Trades', 0):,}")
            st.metric("Win Rate", f"{metrics.get('Win Rate', 0):.2f}%")
        
        with col2:
            st.markdown("#### Profitability")
            st.metric("Net Profit", f"${metrics.get('Net Profit', 0):,.2f}")
            st.metric("Gross Profit", f"${metrics.get('Total Wins', 0):,.2f}")
            st.metric("Gross Loss", f"${metrics.get('Total Losses', 0):,.2f}")
            st.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")
        
        with col3:
            st.markdown("#### Risk Metrics")
            st.metric("Max Drawdown", f"${metrics.get('Max Drawdown', 0):,.2f}")
            st.metric("Max DD %", f"{metrics.get('Max Drawdown %', 0):.2f}%")
            st.metric("Recovery Factor", f"{metrics.get('Recovery Factor', 0):.2f}")
            st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
        
        with col4:
            st.markdown("#### Trade Quality")
            st.metric("Avg Win", f"${metrics.get('Avg Win', 0):,.2f}")
            st.metric("Avg Loss", f"${metrics.get('Avg Loss', 0):,.2f}")
            st.metric("Avg RR Ratio", f"{metrics.get('Avg RR Ratio', 0):.2f}")
            st.metric("Expectancy", f"${metrics.get('Expectancy', 0):.2f}")
        
        st.markdown("---")
        
        # Additional Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Advanced Metrics")
            st.metric("Sortino Ratio", f"{metrics.get('Sortino Ratio', 0):.2f}")
            st.metric("Kelly Criterion", f"{metrics.get('Kelly %', 0):.2f}%")
            st.metric("Max Consecutive Wins", f"{metrics.get('Max Consecutive Wins', 0)}")
            st.metric("Max Consecutive Losses", f"{metrics.get('Max Consecutive Losses', 0)}")
        
        with col2:
            st.markdown("#### 📊 Monthly Performance")
            st.metric("Profitable Months", f"{metrics.get('Profitable Months', 0)} / {metrics.get('Total Months', 0)}")
            st.metric("Monthly Win Rate", f"{metrics.get('Monthly Win Rate', 0):.1f}%")
            st.metric("Largest Win", f"${metrics.get('Largest Win', 0):,.2f}")
            st.metric("Largest Loss", f"${metrics.get('Largest Loss', 0):,.2f}")
    
    # ==================== TAB 3: PORTFOLIO ANALYSIS ====================
    with tabs[2]:
        st.header("Portfolio Analysis")
        
        st.subheader("🔄 Strategy Correlation Matrix")
        
        if len(dataframes) > 1:
            # Create correlation matrix
            strategy_profits = {}
            min_length = min(len(df) for df in dataframes.values())
            
            for name, df in dataframes.items():
                if 'Profit' in df.columns:
                    strategy_profits[name] = df['Profit'].head(min_length).values
            
            corr_df = pd.DataFrame(strategy_profits).corr()
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap='RdYlGn', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                       fmt='.2f', ax=ax)
            ax.set_title('Strategy Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info(f"**Diversification Benefit:** {portfolio_metrics.get('Diversification Benefit', 'Unknown')}")
            st.caption("Lower correlations (closer to 0 or negative) indicate better diversification")
        
        else:
            st.info("Upload multiple strategies to see correlation analysis")
        
        st.markdown("---")
        
        # Combined Performance
        st.subheader("📊 Combined Portfolio Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Portfolio Net Profit", f"${portfolio_metrics.get('Net Profit', 0):,.2f}")
            st.metric("Total Portfolio Trades", f"{portfolio_metrics.get('Total Trades', 0):,}")
        
        with col2:
            st.metric("Portfolio Win Rate", f"{portfolio_metrics.get('Win Rate', 0):.2f}%")
            st.metric("Portfolio Profit Factor", f"{portfolio_metrics.get('Profit Factor', 0):.2f}")
        
        with col3:
            st.metric("Portfolio Sharpe", f"{portfolio_metrics.get('Sharpe Ratio', 0):.2f}")
            st.metric("Portfolio Max DD", f"${portfolio_metrics.get('Max Drawdown', 0):,.2f}")
    
    # ==================== TAB 4: PERFORMANCE COMPARISON ====================
    with tabs[3]:
        st.header("Performance Comparison")
        
        # Create comparison charts
        comparison_metrics = ['Net Profit', 'Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Max Drawdown %']
        
        comparison_data = []
        for name, df in dataframes.items():
            metrics = calculate_advanced_metrics(df)
            comparison_data.append({
                'Strategy': name,
                'Net Profit': metrics.get('Net Profit', 0),
                'Win Rate': metrics.get('Win Rate', 0),
                'Profit Factor': metrics.get('Profit Factor', 0),
                'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
                'Max Drawdown %': abs(metrics.get('Max Drawdown %', 0))
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['Net Profit', 'Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Max Drawdown %']
        colors = plt.cm.Set3(range(len(comp_df)))
        
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            bars = ax.bar(comp_df['Strategy'], comp_df[metric], color=colors)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_xlabel('Strategy')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Hide the last subplot if odd number of metrics
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed comparison table
        st.subheader("Detailed Metrics Comparison")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # ==================== TAB 5: DRAWDOWN ANALYSIS ====================
    with tabs[4]:
        st.header("Drawdown Analysis")
        
        selected_strategy = st.selectbox("Select Strategy for DD Analysis", 
                                        list(dataframes.keys()), key="dd_select")
        df = dataframes[selected_strategy]
        
        if 'Cum. net profit' in df.columns:
            cumulative = df['Cum. net profit']
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max
            drawdown_pct = (drawdown / running_max * 100).fillna(0)
            
            # Plot drawdown
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Equity curve with drawdown
            ax1.plot(df.index, cumulative, label='Equity Curve', linewidth=2, color='#2ecc71')
            ax1.fill_between(df.index, cumulative, running_max, 
                            alpha=0.3, color='#e74c3c', label='Drawdown')
            ax1.plot(df.index, running_max, label='Peak Equity', 
                    linewidth=1, linestyle='--', color='#3498db', alpha=0.7)
            ax1.set_title(f'{selected_strategy} - Equity Curve with Drawdown', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Profit ($)', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown chart
            ax2.fill_between(df.index, drawdown, 0, alpha=0.5, color='#e74c3c')
            ax2.plot(df.index, drawdown, linewidth=1.5, color='#c0392b')
            ax2.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Trade Number', fontsize=12)
            ax2.set_ylabel('Drawdown ($)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=drawdown.min(), color='red', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'Max DD: ${drawdown.min():,.2f}')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Drawdown statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Max Drawdown", f"${drawdown.min():,.2f}")
            with col2:
                st.metric("Max DD %", f"{drawdown_pct.min():.2f}%")
            with col3:
                # Calculate drawdown duration
                in_drawdown = drawdown < 0
                max_dd_duration = 0
                current_duration = 0
                for is_dd in in_drawdown:
                    if is_dd:
                        current_duration += 1
                        max_dd_duration = max(max_dd_duration, current_duration)
                    else:
                        current_duration = 0
                st.metric("Max DD Duration (trades)", max_dd_duration)
            with col4:
                st.metric("Current Drawdown", f"${drawdown.iloc[-1]:,.2f}")
    
    # ==================== TAB 6: TRADE DISTRIBUTION ====================
    with tabs[5]:
        st.header("Trade Distribution Analysis")
        
        selected_strategy = st.selectbox("Select Strategy for Distribution", 
                                        list(dataframes.keys()), key="dist_select")
        df = dataframes[selected_strategy]
        
        # Create distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Day_of_Week' in df.columns:
                st.subheader("Trades by Day of Week")
                fig, ax = plt.subplots(figsize=(8, 5))
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                day_counts = df['Day_of_Week'].value_counts().reindex(day_order, fill_value=0)
                bars = ax.bar(day_counts.index, day_counts.values, color='steelblue')
                ax.set_ylabel('Number of Trades')
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            if 'Month_Year' in df.columns:
                st.subheader("Trades by Month")
                fig, ax = plt.subplots(figsize=(8, 5))
                month_counts = df['Month_Year'].value_counts().sort_index()
                ax.bar(range(len(month_counts)), month_counts.values, color='coral')
                ax.set_xticks(range(len(month_counts)))
                ax.set_xticklabels(month_counts.index, rotation=45)
                ax.set_ylabel('Number of Trades')
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
        
        # Profit distribution
        if 'Profit' in df.columns:
            st.subheader("Profit/Loss Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            returns = df['Profit']
            ax1.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.axvline(returns.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${returns.mean():.2f}')
            ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax1.set_title('Distribution of Trade Returns')
            ax1.set_xlabel('Profit/Loss ($)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(returns, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'),
                       medianprops=dict(color='red', linewidth=2))
            ax2.set_title('Trade Returns Box Plot')
            ax2.set_ylabel('Profit/Loss ($)')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"${returns.mean():.2f}")
            with col2:
                st.metric("Median", f"${returns.median():.2f}")
            with col3:
                st.metric("Std Dev", f"${returns.std():.2f}")
            with col4:
                st.metric("Skewness", f"{returns.skew():.2f}")
        
        # Time in trade analysis
        if 'Time_in_Trade' in df.columns:
            st.subheader("Time in Trade Analysis")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            time_data = df['Time_in_Trade'].dropna()
            ax.hist(time_data, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(time_data.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {time_data.mean():.1f} min')
            ax.set_title('Distribution of Time in Trade')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    # ==================== TAB 7: MONTE CARLO ====================
    with tabs[6]:
        st.header("Monte Carlo Simulation")
        
        if show_monte_carlo:
            selected_strategy = st.selectbox("Select Strategy for Monte Carlo", 
                                            list(dataframes.keys()), key="mc_select")
            df = dataframes[selected_strategy]
            
            if 'Profit' in df.columns:
                st.info(f"Running {n_simulations:,} simulations with {len(df)} trades...")
                
                # Run simulation
                sim_results = monte_carlo_simulation(df, n_simulations=n_simulations)
                
                if sim_results is not None:
                    # Plot results
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Distribution of final P&L
                    ax1.hist(sim_results, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                    ax1.axvline(sim_results.mean(), color='red', linestyle='--', 
                               linewidth=2, label=f'Mean: ${sim_results.mean():,.2f}')
                    ax1.axvline(df['Profit'].sum(), color='green', linestyle='--', 
                               linewidth=2, label=f'Actual: ${df["Profit"].sum():,.2f}')
                    ax1.set_title('Monte Carlo Simulation - Final P&L Distribution')
                    ax1.set_xlabel('Final P&L ($)')
                    ax1.set_ylabel('Frequency')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Percentile plot
                    sorted_results = np.sort(sim_results)
                    percentiles = np.arange(1, 101)
                    percentile_values = np.percentile(sorted_results, percentiles)
                    
                    ax2.plot(percentiles, percentile_values, linewidth=2)
                    ax2.axhline(df['Profit'].sum(), color='green', linestyle='--', 
                               linewidth=2, label='Actual Performance')
                    ax2.fill_between(percentiles, percentile_values, 0, alpha=0.3)
                    ax2.set_title('Monte Carlo Percentile Distribution')
                    ax2.set_xlabel('Percentile')
                    ax2.set_ylabel('P&L ($)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Best Case (95%)", f"${np.percentile(sim_results, 95):,.2f}")
                    with col2:
                        st.metric("Expected (50%)", f"${np.percentile(sim_results, 50):,.2f}")
                    with col3:
                        st.metric("Worst Case (5%)", f"${np.percentile(sim_results, 5):,.2f}")
                    with col4:
                        prob_profit = (sim_results > 0).sum() / len(sim_results) * 100
                        st.metric("Probability of Profit", f"{prob_profit:.1f}%")
                    with col5:
                        st.metric("Actual Result", f"${df['Profit'].sum():,.2f}")
        else:
            st.info("Enable Monte Carlo Simulation in the sidebar to view analysis")
    
    # ==================== TAB 8: DETAILED REPORTS ====================
    with tabs[7]:
        st.header("Detailed Reports")
        
        selected_strategy = st.selectbox("Select Strategy for Report", 
                                        list(dataframes.keys()), key="report_select")
        df = dataframes[selected_strategy]
        metrics = calculate_advanced_metrics(df)
        
        # Generate comprehensive report
        st.subheader(f"📄 Complete Performance Report - {selected_strategy}")
        
        # Trading Activity
        st.markdown("### 📊 Trading Activity")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Total Trades:** {metrics.get('Total Trades', 0):,}")
            st.write(f"**Winning Trades:** {metrics.get('Winning Trades', 0):,}")
            st.write(f"**Losing Trades:** {metrics.get('Losing Trades', 0):,}")
        with col2:
            st.write(f"**Win Rate:** {metrics.get('Win Rate', 0):.2f}%")
            st.write(f"**Monthly Win Rate:** {metrics.get('Monthly Win Rate', 0):.2f}%")
            st.write(f"**Profitable Months:** {metrics.get('Profitable Months', 0)}/{metrics.get('Total Months', 0)}")
        with col3:
            st.write(f"**Max Consecutive Wins:** {metrics.get('Max Consecutive Wins', 0)}")
            st.write(f"**Max Consecutive Losses:** {metrics.get('Max Consecutive Losses', 0)}")
            if 'Time_in_Trade' in df.columns:
                st.write(f"**Avg Time in Trade:** {metrics.get('Avg Time in Trade (min)', 0):.1f} min")
        
        st.markdown("---")
        
        # Profitability
        st.markdown("### 💰 Profitability Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Net Profit:** ${metrics.get('Net Profit', 0):,.2f}")
            st.write(f"**Gross Profit:** ${metrics.get('Total Wins', 0):,.2f}")
            st.write(f"**Gross Loss:** ${metrics.get('Total Losses', 0):,.2f}")
        with col2:
            st.write(f"**Average Win:** ${metrics.get('Avg Win', 0):,.2f}")
            st.write(f"**Average Loss:** ${metrics.get('Avg Loss', 0):,.2f}")
            st.write(f"**Avg RR Ratio:** {metrics.get('Avg RR Ratio', 0):.2f}")
        with col3:
            st.write(f"**Largest Win:** ${metrics.get('Largest Win', 0):,.2f}")
            st.write(f"**Largest Loss:** ${metrics.get('Largest Loss', 0):,.2f}")
            st.write(f"**Profit Factor:** {metrics.get('Profit Factor', 0):.2f}")
        
        st.markdown("---")
        
        # Risk Metrics
        st.markdown("### ⚠️ Risk Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Max Drawdown:** ${metrics.get('Max Drawdown', 0):,.2f}")
            st.write(f"**Max Drawdown %:** {metrics.get('Max Drawdown %', 0):.2f}%")
            st.write(f"**Recovery Factor:** {metrics.get('Recovery Factor', 0):.2f}")
        with col2:
            st.write(f"**Sharpe Ratio:** {metrics.get('Sharpe Ratio', 0):.2f}")
            st.write(f"**Sortino Ratio:** {metrics.get('Sortino Ratio', 0):.2f}")
            st.write(f"**Profit Std Dev:** ${metrics.get('Profit Std Dev', 0):,.2f}")
        with col3:
            st.write(f"**Expectancy:** ${metrics.get('Expectancy', 0):.2f}")
            st.write(f"**Kelly Criterion:** {metrics.get('Kelly %', 0):.2f}%")
            st.write(f"**Peak Profit:** ${metrics.get('Peak Profit', 0):,.2f}")
        
        st.markdown("---")
        
        # Export options
        st.subheader("📥 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export detailed trades
            if st.button("Download Trade History (CSV)"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name=f"{selected_strategy}_trades.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Export metrics report
            if st.button("Download Metrics Report (CSV)"):
                metrics_df = pd.DataFrame([metrics]).T
                metrics_df.columns = ['Value']
                csv = metrics_df.to_csv()
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name=f"{selected_strategy}_metrics.csv",
                    mime="text/csv"
                )
