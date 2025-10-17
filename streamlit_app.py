import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="TradeLens Pro - NinjaTrader Analytics", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def parse_ninjatrader_period_file(file):
    """Parse NinjaTrader period summary CSV files"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        return df
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def clean_currency_column(series):
    """Clean currency columns (remove $, commas, etc.)"""
    return pd.to_numeric(series.astype(str).str.replace('$', '').str.replace(',', '').str.replace('%', ''), errors='coerce')

def process_period_data(df):
    """Process NinjaTrader period summary data"""
    # Parse Period column as datetime
    if 'Period' in df.columns:
        df['Period'] = pd.to_datetime(df['Period'], format='%d/%m/%Y', errors='coerce')
        if df['Period'].isna().any():
            df['Period'] = pd.to_datetime(df['Period'], errors='coerce')
        
        # Extract time features
        df['Month_Year'] = df['Period'].dt.strftime('%Y-%m')
        df['Date'] = df['Period'].dt.date
        df['Year'] = df['Period'].dt.year
        df['Month'] = df['Period'].dt.month
    
    # Clean numeric columns
    numeric_cols = ['Cum. net profit', 'Net profit', 'Gross profit', 'Gross loss', 
                   'Commission', 'Cum. max. drawdown', 'Max. drawdown', 
                   'Avg. trade', 'Avg. winner', 'Avg. loser', 
                   'Lrg. winner', 'Lrg. loser', 'Avg. MAE', 'Avg. MFE', 'Avg. ETD']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_currency_column(df[col])
    
    # Clean percentage columns
    if '% Win' in df.columns:
        df['% Win'] = clean_currency_column(df['% Win'])
    
    if '% Trade' in df.columns:
        df['% Trade'] = clean_currency_column(df['% Trade'])
    
    return df

def calculate_summary_metrics(df):
    """Calculate overall metrics from period data"""
    metrics = {}
    
    if len(df) == 0:
        return metrics
    
    # Use the last row for cumulative metrics
    last_row = df.iloc[-1]
    
    # Basic metrics
    metrics['Net Profit'] = last_row.get('Cum. net profit', 0)
    metrics['Gross Profit'] = df['Gross profit'].sum() if 'Gross profit' in df.columns else 0
    metrics['Gross Loss'] = abs(df['Gross loss'].sum()) if 'Gross loss' in df.columns else 0
    metrics['Total Commission'] = df['Commission'].sum() if 'Commission' in df.columns else 0
    
    # Profit Factor
    if metrics['Gross Loss'] != 0:
        metrics['Profit Factor'] = metrics['Gross Profit'] / metrics['Gross Loss']
    else:
        metrics['Profit Factor'] = 0
    
    # Average metrics
    metrics['Avg Trade'] = df['Avg. trade'].mean() if 'Avg. trade' in df.columns else 0
    metrics['Avg Winner'] = df['Avg. winner'].mean() if 'Avg. winner' in df.columns else 0
    metrics['Avg Loser'] = df['Avg. loser'].mean() if 'Avg. loser' in df.columns else 0
    
    # Win rate (average across periods)
    metrics['Win Rate'] = df['% Win'].mean() if '% Win' in df.columns else 0
    
    # Largest trades
    metrics['Largest Winner'] = df['Lrg. winner'].max() if 'Lrg. winner' in df.columns else 0
    metrics['Largest Loser'] = df['Lrg. loser'].min() if 'Lrg. loser' in df.columns else 0
    
    # Drawdown
    metrics['Max Drawdown'] = last_row.get('Cum. max. drawdown', 0)
    metrics['Peak Profit'] = df['Cum. net profit'].max() if 'Cum. net profit' in df.columns else 0
    
    # Risk metrics
    if 'Net profit' in df.columns:
        returns = df['Net profit']
        metrics['Sharpe Ratio'] = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
        metrics['Sortino Ratio'] = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    # Recovery Factor
    if metrics['Max Drawdown'] != 0:
        metrics['Recovery Factor'] = abs(metrics['Net Profit'] / metrics['Max Drawdown'])
    else:
        metrics['Recovery Factor'] = 0
    
    # MAE/MFE
    metrics['Avg MAE'] = df['Avg. MAE'].mean() if 'Avg. MAE' in df.columns else 0
    metrics['Avg MFE'] = df['Avg. MFE'].mean() if 'Avg. MFE' in df.columns else 0
    
    # Number of periods
    metrics['Total Periods'] = len(df)
    
    # Profitable periods
    if 'Net profit' in df.columns:
        profitable_periods = len(df[df['Net profit'] > 0])
        metrics['Profitable Periods'] = profitable_periods
        metrics['Period Win Rate'] = (profitable_periods / len(df) * 100) if len(df) > 0 else 0
    
    return metrics

def calculate_portfolio_metrics(dataframes, weights=None):
    """Calculate combined portfolio metrics with optional position sizing"""
    if not dataframes:
        return {}, None
    
    # Default to equal weights if not specified
    if weights is None:
        weights = {name: 1.0 for name in dataframes.keys()}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Combine data by Period
    combined_data = []
    
    for name, df in dataframes.items():
        if 'Period' in df.columns and 'Cum. net profit' in df.columns:
            weight = weights.get(name, 1.0)
            temp_df = df[['Period', 'Month_Year', 'Cum. net profit', 'Net profit']].copy()
            # Apply weight to returns
            temp_df['Cum. net profit'] = temp_df['Cum. net profit'] * weight
            temp_df['Net profit'] = temp_df['Net profit'] * weight
            temp_df.columns = ['Period', 'Month_Year', f'{name}_Cum', f'{name}_Net']
            combined_data.append(temp_df)
    
    if not combined_data:
        return {}, None
    
    # Merge all dataframes on Period
    combined = combined_data[0]
    for df in combined_data[1:]:
        combined = pd.merge(combined, df, on=['Period', 'Month_Year'], how='outer')
    
    combined = combined.sort_values('Period').fillna(method='ffill').fillna(0)
    
    # Calculate combined cumulative profit
    cum_cols = [col for col in combined.columns if '_Cum' in col]
    combined['Portfolio_Cum'] = combined[cum_cols].sum(axis=1)
    
    # Calculate combined net profit per period
    net_cols = [col for col in combined.columns if '_Net' in col]
    combined['Portfolio_Net'] = combined[net_cols].sum(axis=1)
    
    # Calculate correlations
    correlations = {}
    strategy_names = list(dataframes.keys())
    
    for i, name1 in enumerate(strategy_names):
        for name2 in strategy_names[i+1:]:
            col1 = f'{name1}_Net'
            col2 = f'{name2}_Net'
            if col1 in combined.columns and col2 in combined.columns:
                corr = combined[col1].corr(combined[col2])
                correlations[f"{name1} vs {name2}"] = corr
    
    # Portfolio summary metrics
    portfolio_metrics = {}
    portfolio_metrics['Net Profit'] = combined['Portfolio_Cum'].iloc[-1]
    portfolio_metrics['Peak Profit'] = combined['Portfolio_Cum'].max()
    
    # Drawdown
    cumulative = combined['Portfolio_Cum']
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    portfolio_metrics['Max Drawdown'] = drawdown.min()
    portfolio_metrics['Max Drawdown %'] = (drawdown / running_max * 100).min() if running_max.max() > 0 else 0
    
    # Risk metrics
    returns = combined['Portfolio_Net']
    portfolio_metrics['Sharpe Ratio'] = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
    portfolio_metrics['Sortino Ratio'] = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    portfolio_metrics['Strategy Correlations'] = correlations
    portfolio_metrics['Number of Strategies'] = len(dataframes)
    
    # Diversification benefit
    if correlations:
        avg_corr = np.mean(list(correlations.values()))
        if avg_corr < 0.3:
            portfolio_metrics['Diversification Benefit'] = 'Excellent'
        elif avg_corr < 0.5:
            portfolio_metrics['Diversification Benefit'] = 'Good'
        elif avg_corr < 0.7:
            portfolio_metrics['Diversification Benefit'] = 'Fair'
        else:
            portfolio_metrics['Diversification Benefit'] = 'Limited'
    
    return portfolio_metrics, combined

# ==================== STREAMLIT APP ====================

st.markdown('<h1 class="main-header">🎯 TradeLens Pro - NinjaTrader Analytics</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=TradeLens+Pro", use_container_width=True)
    st.markdown("---")
    
    st.header("📁 Upload Trading Data")
    st.markdown("*NinjaTrader Period Summary Reports*")
    
    uploaded_files = st.file_uploader(
        "Upload CSV/Excel Files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload NinjaTrader performance summary files"
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
        
        # Show combined account info
        if len(uploaded_files) > 1:
            st.success(f"✅ {len(uploaded_files)} strategies loaded")
            st.info("💼 Combined Account analysis available!")
        else:
            st.info(f"📊 {len(uploaded_files)} strategy loaded")
            st.caption("Upload 2+ strategies for portfolio analysis")
    
    st.markdown("---")
    
    # Analysis Options
    st.subheader("⚙️ Analysis Options")
    
    # Position sizing
    if uploaded_files and len(uploaded_files) > 1:
        with st.expander("💰 Position Sizing / Allocation"):
            st.caption("Adjust capital allocation to each strategy")
            
            allocation_mode = st.radio(
                "Allocation Mode",
                ["Equal Weight", "Custom Weight"],
                help="Equal: Same capital to each strategy. Custom: Set your own allocation"
            )
            
            if allocation_mode == "Custom Weight":
                st.caption("Set allocation % for each strategy (must sum to 100%)")
                # This will be used in processing
                st.session_state['custom_allocation'] = True
                
                if 'strategy_weights' not in st.session_state:
                    st.session_state['strategy_weights'] = {}
                
                # We'll populate this after files are loaded
            else:
                st.session_state['custom_allocation'] = False
    
    show_benchmark = st.checkbox("Add Buy & Hold Benchmark", value=False)
    if show_benchmark:
        benchmark_ticker = st.text_input("Ticker Symbol", value="ES=F", help="e.g., ES=F, NQ=F, SPY")
        benchmark_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
    
    st.markdown("---")
    st.caption("Built for NinjaTrader | Version 2.0")

# Main Content
if not uploaded_files:
    st.info("👆 Upload your NinjaTrader performance summary files to begin analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Individual Analytics")
        st.markdown("- Period-by-period performance")
        st.markdown("- Risk-adjusted returns")
        st.markdown("- Drawdown analysis")
    
    with col2:
        st.markdown("### 🔄 Portfolio Analytics")
        st.markdown("- Combined strategy performance")
        st.markdown("- Strategy correlation analysis")
        st.markdown("- Diversification benefits")
    
    with col3:
        st.markdown("### 📈 Benchmarking")
        st.markdown("- Compare vs Buy & Hold")
        st.markdown("- Monthly equity curves")
        st.markdown("- Performance comparison")
    
    st.markdown("---")
    st.markdown("### 📋 NinjaTrader Export Instructions")
    st.markdown("""
    1. In NinjaTrader, go to **Performance** tab
    2. Set your desired time period
    3. Right-click and select **Export**
    4. Save as CSV
    5. Upload here for instant analytics
    """)

else:
    # Load and process all files
    dataframes = {}
    
    for i, file in enumerate(uploaded_files):
        df = parse_ninjatrader_period_file(file)
        if df is not None:
            strategy_name = strategy_names.get(i, f"Strategy {i+1}")
            df = process_period_data(df)
            dataframes[strategy_name] = df
    
    if not dataframes:
        st.error("No valid data found in uploaded files. Please check your file format.")
        st.stop()
    
    # Handle custom allocation weights
    allocation_weights = None
    if st.session_state.get('custom_allocation', False):
        with st.sidebar.expander("💰 Position Sizing / Allocation", expanded=True):
            st.caption("Set allocation % for each strategy (must sum to 100%)")
            
            weights = {}
            total = 0
            
            for name in dataframes.keys():
                weight = st.number_input(
                    f"{name} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0 / len(dataframes),
                    step=5.0,
                    key=f"weight_{name}"
                )
                weights[name] = weight / 100.0  # Convert to decimal
                total += weight
            
            if abs(total - 100) > 0.01:
                st.warning(f"⚠️ Total allocation: {total:.1f}% (should be 100%)")
            else:
                st.success(f"✅ Total allocation: {total:.1f}%")
                allocation_weights = weights
    
    # Create tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "💼 Combined Account",
        "🎯 Individual Strategies",
        "🔄 Portfolio Analysis",
        "📈 Equity Curves",
        "📉 Drawdown Analysis",
        "📊 Period Analysis"
    ])
    
    # ==================== TAB 1: DASHBOARD ====================
    with tabs[0]:
        st.header("Performance Dashboard")
        
        # Calculate portfolio metrics with weights
        portfolio_metrics, combined_df = calculate_portfolio_metrics(dataframes, allocation_weights)
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Portfolio Net Profit",
                f"${portfolio_metrics.get('Net Profit', 0):,.2f}",
                delta=f"{len(dataframes)} Strategies"
            )
        
        with col2:
            st.metric(
                "Peak Profit",
                f"${portfolio_metrics.get('Peak Profit', 0):,.2f}"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{portfolio_metrics.get('Sharpe Ratio', 0):.2f}",
                delta="Excellent" if portfolio_metrics.get('Sharpe Ratio', 0) > 1 else "Fair"
            )
        
        with col4:
            st.metric(
                "Sortino Ratio",
                f"{portfolio_metrics.get('Sortino Ratio', 0):.2f}"
            )
        
        with col5:
            st.metric(
                "Max Drawdown",
                f"${portfolio_metrics.get('Max Drawdown', 0):,.2f}",
                delta=f"{portfolio_metrics.get('Max Drawdown %', 0):.1f}%",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Strategy Performance Summary
        st.subheader("Strategy Performance Summary")
        summary_data = []
        for name, df in dataframes.items():
            metrics = calculate_summary_metrics(df)
            summary_data.append({
                'Strategy': name,
                'Net Profit': f"${metrics.get('Net Profit', 0):,.2f}",
                'Win Rate': f"{metrics.get('Win Rate', 0):.1f}%",
                'Profit Factor': f"{metrics.get('Profit Factor', 0):.2f}",
                'Sharpe': f"{metrics.get('Sharpe Ratio', 0):.2f}",
                'Max DD': f"${metrics.get('Max Drawdown', 0):,.2f}",
                'Periods': metrics.get('Total Periods', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # ==================== TAB 2: COMBINED ACCOUNT ====================
    with tabs[1]:
        st.header("💼 Combined Account View")
        
        # Show allocation if custom
        if allocation_weights:
            st.info("🎯 Using custom allocation weights")
            weight_display = " | ".join([f"{name}: {w*100:.0f}%" for name, w in allocation_weights.items()])
            st.caption(weight_display)
        else:
            st.info("📊 Using equal weight allocation across all strategies")
        
        st.markdown("---")
        
        if combined_df is not None and len(dataframes) > 1:
            
            # Quick comparison box
            st.subheader("🎯 Combined vs Individual - Quick Comparison")
            
            # Calculate averages for individual strategies
            individual_metrics = []
            for name, df in dataframes.items():
                metrics = calculate_summary_metrics(df)
                individual_metrics.append(metrics)
            
            avg_individual_profit = np.mean([m.get('Net Profit', 0) for m in individual_metrics])
            avg_individual_sharpe = np.mean([m.get('Sharpe Ratio', 0) for m in individual_metrics])
            avg_individual_dd = np.mean([abs(m.get('Max Drawdown', 0)) for m in individual_metrics])
            
            combined_profit = portfolio_metrics.get('Net Profit', 0)
            combined_sharpe = portfolio_metrics.get('Sharpe Ratio', 0)
            combined_dd = abs(portfolio_metrics.get('Max Drawdown', 0))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Combined Net Profit",
                    f"${combined_profit:,.2f}",
                    delta=f"vs ${avg_individual_profit:,.2f} avg individual"
                )
            
            with col2:
                sharpe_improvement = ((combined_sharpe - avg_individual_sharpe) / avg_individual_sharpe * 100) if avg_individual_sharpe != 0 else 0
                st.metric(
                    "Combined Sharpe Ratio",
                    f"{combined_sharpe:.2f}",
                    delta=f"{sharpe_improvement:+.1f}% vs individual",
                    delta_color="normal" if sharpe_improvement > 0 else "inverse"
                )
            
            with col3:
                dd_improvement = ((avg_individual_dd - combined_dd) / avg_individual_dd * 100) if avg_individual_dd != 0 else 0
                st.metric(
                    "Combined Max DD",
                    f"${combined_dd:,.2f}",
                    delta=f"{dd_improvement:+.1f}% better than avg",
                    delta_color="normal" if dd_improvement > 0 else "inverse"
                )
            
            with col4:
                diversification = portfolio_metrics.get('Diversification Benefit', 'Unknown')
                color_map = {'Excellent': '🟢', 'Good': '🟡', 'Fair': '🟠', 'Limited': '🔴'}
                st.metric(
                    "Diversification",
                    f"{color_map.get(diversification, '⚪')} {diversification}",
                )
            
            st.markdown("---")
            
            # Combined Period-by-Period Performance
            st.subheader("Period-by-Period Combined Performance")
            
            # Create period performance chart
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Bar chart of period returns
            colors = ['green' if x > 0 else 'red' for x in combined_df['Portfolio_Net']]
            ax.bar(range(len(combined_df)), combined_df['Portfolio_Net'], color=colors, alpha=0.6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_title('Combined Account - Period Returns', fontsize=14, fontweight='bold')
            ax.set_xlabel('Period Number', fontsize=12)
            ax.set_ylabel('Net Profit ($)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Combined equity curve with individual contributions
            st.subheader("Combined Account Equity Curve")
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot individual strategies in background
            cum_cols = [col for col in combined_df.columns if '_Cum' in col and col != 'Portfolio_Cum']
            for col in cum_cols:
                strategy_name = col.replace('_Cum', '')
                ax.plot(combined_df.index, combined_df[col], label=strategy_name, 
                       linewidth=1, alpha=0.4, linestyle='--')
            
            # Plot combined portfolio prominently
            ax.plot(combined_df.index, combined_df['Portfolio_Cum'], 
                   label='Combined Account', linewidth=3, color='#2ecc71')
            
            ax.set_title('Combined Account Equity Curve (All Strategies)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Period', fontsize=12)
            ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Strategy contribution breakdown
            st.subheader("Strategy Contribution to Combined Account")
            
            # Calculate each strategy's contribution
            contributions = {}
            for col in cum_cols:
                strategy_name = col.replace('_Cum', '')
                final_value = combined_df[col].iloc[-1]
                contributions[strategy_name] = final_value
            
            # Create pie chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Pie chart of final contributions
            colors_pie = plt.cm.Set3(range(len(contributions)))
            wedges, texts, autotexts = ax1.pie(contributions.values(), 
                                               labels=contributions.keys(),
                                               autopct='%1.1f%%',
                                               colors=colors_pie,
                                               startangle=90)
            ax1.set_title('Profit Contribution by Strategy', fontsize=12, fontweight='bold')
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
            
            # Bar chart of contributions
            ax2.bar(contributions.keys(), contributions.values(), color=colors_pie)
            ax2.set_title('Final Profit Contribution ($)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Profit ($)')
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Combined account statistics
            st.subheader("Combined Account Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_periods = len(combined_df)
                profitable_periods = len(combined_df[combined_df['Portfolio_Net'] > 0])
                st.metric("Total Periods", total_periods)
                st.metric("Profitable Periods", profitable_periods)
                st.metric("Period Win Rate", f"{(profitable_periods/total_periods*100):.1f}%")
            
            with col2:
                avg_period_return = combined_df['Portfolio_Net'].mean()
                median_period_return = combined_df['Portfolio_Net'].median()
                std_period_return = combined_df['Portfolio_Net'].std()
                st.metric("Avg Period Return", f"${avg_period_return:,.2f}")
                st.metric("Median Period Return", f"${median_period_return:,.2f}")
                st.metric("Std Dev", f"${std_period_return:,.2f}")
            
            with col3:
                best_period = combined_df['Portfolio_Net'].max()
                worst_period = combined_df['Portfolio_Net'].min()
                st.metric("Best Period", f"${best_period:,.2f}")
                st.metric("Worst Period", f"${worst_period:,.2f}")
                st.metric("Best/Worst Ratio", f"{abs(best_period/worst_period):.2f}")
            
            with col4:
                # Calculate streak statistics
                wins = (combined_df['Portfolio_Net'] > 0).astype(int)
                winning_streaks = []
                losing_streaks = []
                current_streak = 0
                current_type = None
                
                for win in wins:
                    if win == 1:
                        if current_type == 'win':
                            current_streak += 1
                        else:
                            if current_type == 'loss' and current_streak > 0:
                                losing_streaks.append(current_streak)
                            current_streak = 1
                            current_type = 'win'
                    else:
                        if current_type == 'loss':
                            current_streak += 1
                        else:
                            if current_type == 'win' and current_streak > 0:
                                winning_streaks.append(current_streak)
                            current_streak = 1
                            current_type = 'loss'
                
                max_win_streak = max(winning_streaks) if winning_streaks else 0
                max_loss_streak = max(losing_streaks) if losing_streaks else 0
                
                st.metric("Max Win Streak", f"{max_win_streak} periods")
                st.metric("Max Loss Streak", f"{max_loss_streak} periods")
                st.metric("Current Streak", f"{'Win' if wins.iloc[-1] else 'Loss'}")
            
            st.markdown("---")
            
            # Strategy correlation impact
            st.subheader("📊 Diversification Benefits")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Why Trade Multiple Strategies?")
                
                # Calculate individual strategy stats
                individual_sharpes = []
                individual_max_dds = []
                for name, df in dataframes.items():
                    metrics = calculate_summary_metrics(df)
                    individual_sharpes.append(metrics.get('Sharpe Ratio', 0))
                    individual_max_dds.append(abs(metrics.get('Max Drawdown', 0)))
                
                avg_individual_sharpe = np.mean(individual_sharpes)
                portfolio_sharpe = portfolio_metrics.get('Sharpe Ratio', 0)
                
                avg_individual_dd = np.mean(individual_max_dds)
                portfolio_dd = abs(portfolio_metrics.get('Max Drawdown', 0))
                
                st.write(f"**Average Individual Sharpe:** {avg_individual_sharpe:.2f}")
                st.write(f"**Combined Portfolio Sharpe:** {portfolio_sharpe:.2f}")
                
                if portfolio_sharpe > avg_individual_sharpe:
                    improvement = ((portfolio_sharpe - avg_individual_sharpe) / avg_individual_sharpe * 100)
                    st.success(f"✅ {improvement:.1f}% improvement in risk-adjusted returns!")
                
                st.write(f"**Average Individual Max DD:** ${avg_individual_dd:,.2f}")
                st.write(f"**Combined Portfolio Max DD:** ${portfolio_dd:,.2f}")
                
                if portfolio_dd < avg_individual_dd:
                    improvement = ((avg_individual_dd - portfolio_dd) / avg_individual_dd * 100)
                    st.success(f"✅ {improvement:.1f}% reduction in maximum drawdown!")
            
            with col2:
                st.markdown("#### Portfolio Benefits Summary")
                
                diversification = portfolio_metrics.get('Diversification Benefit', 'Unknown')
                
                if diversification == 'Excellent':
                    st.success("🌟 **Excellent Diversification**")
                    st.write("Your strategies have low correlation, providing strong diversification benefits.")
                elif diversification == 'Good':
                    st.info("✅ **Good Diversification**")
                    st.write("Your strategies complement each other well.")
                elif diversification == 'Fair':
                    st.warning("⚠️ **Fair Diversification**")
                    st.write("Some correlation between strategies. Consider adding uncorrelated strategies.")
                else:
                    st.error("❌ **Limited Diversification**")
                    st.write("High correlation between strategies reduces diversification benefits.")
                
                st.markdown("---")
                
                st.write("**Key Benefits of This Portfolio:**")
                st.write("✓ Smoother equity curve")
                st.write("✓ Reduced drawdown risk")
                st.write("✓ More consistent returns")
                st.write("✓ Better risk-adjusted performance")
            
            st.markdown("---")
            
            # Rolling performance metrics
            st.subheader("📈 Rolling Performance Analysis")
            
            window = st.slider("Rolling Window (periods)", 10, 50, 20)
            
            # Calculate rolling metrics
            rolling_sharpe = []
            rolling_win_rate = []
            
            for i in range(window, len(combined_df)):
                window_data = combined_df.iloc[i-window:i]['Portfolio_Net']
                
                # Rolling Sharpe
                if window_data.std() > 0:
                    sharpe = (window_data.mean() / window_data.std() * np.sqrt(252))
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0)
                
                # Rolling win rate
                win_rate = (window_data > 0).sum() / len(window_data) * 100
                rolling_win_rate.append(win_rate)
            
            # Plot rolling metrics
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            x_vals = range(window, len(combined_df))
            
            ax1.plot(x_vals, rolling_sharpe, linewidth=2, color='#3498db')
            ax1.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (>1)')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax1.set_title(f'Rolling Sharpe Ratio ({window}-period window)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(x_vals, rolling_win_rate, linewidth=2, color='#e74c3c')
            ax2.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50%')
            ax2.set_title(f'Rolling Win Rate ({window}-period window)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Period')
            ax2.set_ylabel('Win Rate (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif len(dataframes) == 1:
            st.warning("📤 Upload multiple strategy files to see combined account analysis")
            st.info("Upload 2 or more strategies to see how they work together in one account!")
        else:
            st.error("No data available")
    
    # ==================== TAB 3: INDIVIDUAL STRATEGIES ====================
    with tabs[2]:
        st.header("Individual Strategy Analysis")
        
        selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()))
        df = dataframes[selected_strategy]
        metrics = calculate_summary_metrics(df)
        
        # Metrics Grid
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### Profitability")
            st.metric("Net Profit", f"${metrics.get('Net Profit', 0):,.2f}")
            st.metric("Gross Profit", f"${metrics.get('Gross Profit', 0):,.2f}")
            st.metric("Gross Loss", f"${metrics.get('Gross Loss', 0):,.2f}")
            st.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")
        
        with col2:
            st.markdown("#### Trade Quality")
            st.metric("Win Rate", f"{metrics.get('Win Rate', 0):.2f}%")
            st.metric("Avg Trade", f"${metrics.get('Avg Trade', 0):,.2f}")
            st.metric("Avg Winner", f"${metrics.get('Avg Winner', 0):,.2f}")
            st.metric("Avg Loser", f"${metrics.get('Avg Loser', 0):,.2f}")
        
        with col3:
            st.markdown("#### Risk Metrics")
            st.metric("Max Drawdown", f"${metrics.get('Max Drawdown', 0):,.2f}")
            st.metric("Peak Profit", f"${metrics.get('Peak Profit', 0):,.2f}")
            st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
            st.metric("Sortino Ratio", f"{metrics.get('Sortino Ratio', 0):.2f}")
        
        with col4:
            st.markdown("#### Trade Stats")
            st.metric("Largest Winner", f"${metrics.get('Largest Winner', 0):,.2f}")
            st.metric("Largest Loser", f"${metrics.get('Largest Loser', 0):,.2f}")
            st.metric("Avg MAE", f"${metrics.get('Avg MAE', 0):,.2f}")
            st.metric("Avg MFE", f"${metrics.get('Avg MFE', 0):,.2f}")
        
        st.markdown("---")
        
        # Period Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Period Statistics")
            st.metric("Total Periods", f"{metrics.get('Total Periods', 0)}")
            st.metric("Profitable Periods", f"{metrics.get('Profitable Periods', 0)}")
            st.metric("Period Win Rate", f"{metrics.get('Period Win Rate', 0):.1f}%")
        
        with col2:
            st.markdown("#### Risk/Reward")
            st.metric("Recovery Factor", f"{metrics.get('Recovery Factor', 0):.2f}")
            st.metric("Total Commission", f"${metrics.get('Total Commission', 0):,.2f}")
    
    # ==================== TAB 4: PORTFOLIO ANALYSIS ====================
    with tabs[3]:
        st.header("Portfolio Analysis")
        
        st.subheader("🔄 Strategy Correlation Matrix")
        
        if len(dataframes) > 1 and combined_df is not None:
            # Create correlation matrix from net profit columns
            net_cols = [col for col in combined_df.columns if '_Net' in col and col != 'Portfolio_Net']
            
            if len(net_cols) > 1:
                corr_data = combined_df[net_cols].copy()
                corr_data.columns = [col.replace('_Net', '') for col in corr_data.columns]
                corr_matrix = corr_data.corr()
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, 
                           square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                           fmt='.2f', ax=ax, vmin=-1, vmax=1)
                ax.set_title('Strategy Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info(f"**Diversification Benefit:** {portfolio_metrics.get('Diversification Benefit', 'Unknown')}")
                st.caption("Lower correlations (closer to 0 or negative) indicate better diversification")
        else:
            st.info("Upload multiple strategies to see correlation analysis")
        
        st.markdown("---")
        
        # Combined Performance
        st.subheader("📊 Portfolio Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Portfolio Net Profit", f"${portfolio_metrics.get('Net Profit', 0):,.2f}")
            st.metric("Portfolio Peak Profit", f"${portfolio_metrics.get('Peak Profit', 0):,.2f}")
        
        with col2:
            st.metric("Portfolio Sharpe", f"{portfolio_metrics.get('Sharpe Ratio', 0):.2f}")
            st.metric("Portfolio Sortino", f"{portfolio_metrics.get('Sortino Ratio', 0):.2f}")
        
        with col3:
            st.metric("Portfolio Max DD", f"${portfolio_metrics.get('Max Drawdown', 0):,.2f}")
            st.metric("Portfolio Max DD %", f"{portfolio_metrics.get('Max Drawdown %', 0):.2f}%")
    
    # ==================== TAB 5: EQUITY CURVES ====================
    with tabs[4]:
        st.header("Equity Curves Comparison")
        
        if combined_df is not None:
            # Aggregate to monthly data
            monthly_data = {}
            
            for name, df in dataframes.items():
                if 'Month_Year' in df.columns and 'Cum. net profit' in df.columns:
                    monthly = df.groupby('Month_Year')['Cum. net profit'].last().sort_index()
                    monthly_data[name] = monthly
            
            if monthly_data:
                monthly_combined = pd.DataFrame(monthly_data)
                monthly_combined = monthly_combined.ffill().fillna(0)
                monthly_combined['Portfolio'] = monthly_combined.sum(axis=1)
                
                # Add benchmark if selected
                if show_benchmark and benchmark_ticker:
                    try:
                        with st.spinner(f"Fetching {benchmark_ticker} data..."):
                            start_date = monthly_combined.index.min() + '-01'
                            end_date = monthly_combined.index.max() + '-28'
                            
                            market_data = yf.download(benchmark_ticker, start=start_date, 
                                                    end=end_date, interval="1mo", progress=False)
                            
                            if not market_data.empty:
                                market_data['Month_Year'] = market_data.index.strftime('%Y-%m')
                                monthly_prices = market_data.groupby('Month_Year')['Close'].last()
                                
                                first_price = monthly_prices.iloc[0]
                                buy_hold_returns = ((monthly_prices / first_price) - 1) * benchmark_capital
                                monthly_combined['Buy & Hold'] = buy_hold_returns.reindex(monthly_combined.index).ffill().fillna(0)
                    except Exception as e:
                        st.warning(f"Could not fetch benchmark data: {e}")
                
                # Plot
                fig, ax = plt.subplots(figsize=(14, 7))
                
                for col in monthly_combined.columns:
                    if col == 'Portfolio':
                        ax.plot(monthly_combined.index, monthly_combined[col], 
                               label=col, linewidth=2.5, color='#2ecc71')
                    elif col == 'Buy & Hold':
                        ax.plot(monthly_combined.index, monthly_combined[col], 
                               label=col, linewidth=2, linestyle='--', color='#e74c3c')
                    else:
                        ax.plot(monthly_combined.index, monthly_combined[col], 
                               label=col, linewidth=1.5, alpha=0.7)
                
                ax.set_title('Monthly Cumulative Profit - Comparison', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('Month-Year', fontsize=12)
                ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
                
                # Format x-axis
                x_ticks = range(0, len(monthly_combined.index), max(1, len(monthly_combined.index) // 12))
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([monthly_combined.index[i] for i in x_ticks], rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary stats
                st.subheader("Monthly Performance Summary")
                final_values = monthly_combined.iloc[-1]
                
                cols = st.columns(len(monthly_combined.columns))
                for i, col in enumerate(monthly_combined.columns):
                    with cols[i]:
                        st.metric(col, f"${final_values[col]:,.2f}")
    
    # ==================== TAB 6: DRAWDOWN ANALYSIS ====================
    with tabs[5]:
        st.header("Drawdown Analysis")
        
        selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="dd_select")
        df = dataframes[selected_strategy]
        
        if 'Cum. net profit' in df.columns:
            cumulative = df['Cum. net profit']
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max
            drawdown_pct = (drawdown / running_max * 100).fillna(0)
            
            # Plot
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
            ax2.set_xlabel('Period', fontsize=12)
            ax2.set_ylabel('Drawdown ($)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=drawdown.min(), color='red', linestyle='--', 
                       linewidth=1, alpha=0.7, label=f'Max DD: ${drawdown.min():,.2f}')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Max Drawdown", f"${drawdown.min():,.2f}")
            with col2:
                st.metric("Max DD %", f"{drawdown_pct.min():.2f}%")
            with col3:
                st.metric("Current Drawdown", f"${drawdown.iloc[-1]:,.2f}")
            with col4:
                in_drawdown = drawdown < 0
                current_dd_duration = 0
                for is_dd in reversed(in_drawdown.values):
                    if is_dd:
                        current_dd_duration += 1
                    else:
                        break
                st.metric("Current DD Duration", f"{current_dd_duration} periods")
    
    # ==================== TAB 7: PERIOD ANALYSIS ====================
    with tabs[6]:
        st.header("Period-by-Period Analysis")
        
        selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="period_select")
        df = dataframes[selected_strategy]
        
        # Period performance distribution
        if 'Net profit' in df.columns:
            st.subheader("Period Returns Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            returns = df['Net profit']
            
            # Histogram
            ax1.hist(returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.axvline(returns.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${returns.mean():.2f}')
            ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax1.set_title('Distribution of Period Returns')
            ax1.set_xlabel('Net Profit ($)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(returns, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'),
                       medianprops=dict(color='red', linewidth=2))
            ax2.set_title('Period Returns Box Plot')
            ax2.set_ylabel('Net Profit ($)')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Return", f"${returns.mean():.2f}")
            with col2:
                st.metric("Median Return", f"${returns.median():.2f}")
            with col3:
                st.metric("Std Dev", f"${returns.std():.2f}")
            with col4:
                profitable = len(returns[returns > 0])
                st.metric("Profitable Periods", f"{profitable}/{len(returns)}")
        
        # Show data table
        st.subheader("Period Data")
        display_cols = ['Period', 'Cum. net profit', 'Net profit', '% Win', 'Avg. trade', 
                       'Max. drawdown', 'Gross profit', 'Gross loss']
        display_cols = [col for col in display_cols if col in df.columns]
        
        st.dataframe(df[display_cols].tail(20), use_container_width=True)
        
        # Export
        if st.button("Download Full Period Data (CSV)"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name=f"{selected_strategy}_periods.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.caption("TradeLens Pro v2.0 | Built for NinjaTrader Period Reports")
