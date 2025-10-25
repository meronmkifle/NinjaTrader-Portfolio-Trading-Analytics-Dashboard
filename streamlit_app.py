import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf

st.set_page_config(
    page_title="NinjaTrader Trading Analytics Dashboard",
    page_icon=None, 
    layout="wide"
)

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

def detect_format_type(df):
    """Detect the type of NinjaTrader export format"""
    if 'Period' not in df.columns:
        return 'unknown'
    
    # Check first non-null Period value
    first_period = df['Period'].dropna().iloc[0] if len(df['Period'].dropna()) > 0 else ''
    
    # Check if it's day of week format
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if any(day in str(first_period) for day in days_of_week):
        return 'day_of_week'
    
    # Check if it has time component (trades format)
    if ':' in str(first_period) and ('AM' in str(first_period) or 'PM' in str(first_period)):
        return 'trades'
    
    # Check if Cum. net profit has n/a values
    if 'Cum. net profit' in df.columns:
        has_na = df['Cum. net profit'].astype(str).str.contains('n/a', case=False).any()
        if has_na:
            return 'day_of_week'  # or some other aggregated format
    
    # Otherwise it's a regular period format (daily, weekly, monthly, yearly)
    return 'period'

def clean_currency_column(series):
    """Clean currency columns (remove $, commas, handle n/a)"""
    # Convert to string and handle n/a
    cleaned = series.astype(str).str.replace('$', '').str.replace(',', '').str.replace('%', '')
    # Replace n/a with NaN
    cleaned = cleaned.replace(['n/a', 'N/A', 'na', 'NA'], np.nan)
    return pd.to_numeric(cleaned, errors='coerce')

def parse_period_column(df, format_type):
    """Parse Period column based on format type"""
    if format_type == 'day_of_week':
        # Keep day names as-is, create a numeric index for ordering
        day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df['Day_Order'] = df['Period'].map(day_order)
        df['Period_Original'] = df['Period']
        return df
    
    elif format_type == 'trades':
        # Parse datetime with time component
        try:
            # Try format: DD/MM/YYYY HH:MM AM/PM
            df['Period'] = pd.to_datetime(df['Period'], format='%d/%m/%Y %I:%M %p', errors='coerce')
        except:
            try:
                df['Period'] = pd.to_datetime(df['Period'], format='%d/%m/%Y %H:%M', errors='coerce')
            except:
                df['Period'] = pd.to_datetime(df['Period'], errors='coerce')
        
        if 'Period' in df.columns and not df['Period'].isna().all():
            df['Date'] = df['Period'].dt.date
            df['Year'] = df['Period'].dt.year
            df['Month'] = df['Period'].dt.month
            df['Month_Year'] = df['Period'].dt.strftime('%Y-%m')
            df['Time'] = df['Period'].dt.time
        return df
    
    else:  # period format (daily, weekly, monthly, yearly)
        # Try different date formats
        try:
            df['Period'] = pd.to_datetime(df['Period'], format='%d/%m/%Y', errors='coerce')
        except:
            try:
                df['Period'] = pd.to_datetime(df['Period'], format='%m/%d/%Y', errors='coerce')
            except:
                df['Period'] = pd.to_datetime(df['Period'], errors='coerce')
        
        if 'Period' in df.columns and not df['Period'].isna().all():
            df['Month_Year'] = df['Period'].dt.strftime('%Y-%m')
            df['Date'] = df['Period'].dt.date
            df['Year'] = df['Period'].dt.year
            df['Month'] = df['Period'].dt.month
            df['Day_of_Week'] = df['Period'].dt.day_name()
        
        return df

def parse_ninjatrader_file(file):
    """Parse NinjaTrader CSV files with format auto-detection"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Detect format type
        format_type = detect_format_type(df)
        
        # Parse Period column based on format
        df = parse_period_column(df, format_type)
        
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
        
        # Store format type as metadata
        df.attrs['format_type'] = format_type
        
        return df
    
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def calculate_summary_metrics(df):
    """Calculate overall metrics from period data - handles all formats"""
    metrics = {}
    format_type = df.attrs.get('format_type', 'period')
    
    if len(df) == 0:
        return metrics
    
    # For day_of_week or other aggregated formats, we may not have cumulative data
    if format_type == 'day_of_week':
        # Calculate from available data
        metrics['Net Profit'] = df['Net profit'].sum() if 'Net profit' in df.columns else 0
        metrics['Gross Profit'] = df['Gross profit'].sum() if 'Gross profit' in df.columns else 0
        metrics['Gross Loss'] = abs(df['Gross loss'].sum()) if 'Gross loss' in df.columns else 0
        metrics['Total Commission'] = df['Commission'].sum() if 'Commission' in df.columns else 0
    else:
        # Use the last row for cumulative metrics
        last_row = df.iloc[-1]
        metrics['Net Profit'] = last_row.get('Cum. net profit', 0)
        if pd.isna(metrics['Net Profit']):
            metrics['Net Profit'] = df['Net profit'].sum() if 'Net profit' in df.columns else 0
        
        metrics['Gross Profit'] = df['Gross profit'].sum() if 'Gross profit' in df.columns else 0
        metrics['Gross Loss'] = abs(df['Gross loss'].sum()) if 'Gross loss' in df.columns else 0
        metrics['Total Commission'] = df['Commission'].sum() if 'Commission' in df.columns else 0
    
    # Profit Factor
    if metrics['Gross Loss'] != 0:
        metrics['Profit Factor'] = metrics['Gross Profit'] / metrics['Gross Loss']
    else:
        metrics['Profit Factor'] = 0
    
    # Average metrics (use mean where available, handle NaN)
    metrics['Avg Trade'] = df['Avg. trade'].mean() if 'Avg. trade' in df.columns else 0
    metrics['Avg Winner'] = df['Avg. winner'].mean() if 'Avg. winner' in df.columns else 0
    metrics['Avg Loser'] = df['Avg. loser'].mean() if 'Avg. loser' in df.columns else 0
    
    # Win rate (average across periods)
    metrics['Win Rate'] = df['% Win'].mean() if '% Win' in df.columns else 0
    
    # Largest trades
    metrics['Largest Winner'] = df['Lrg. winner'].max() if 'Lrg. winner' in df.columns else 0
    metrics['Largest Loser'] = df['Lrg. loser'].min() if 'Lrg. loser' in df.columns else 0
    
    # Drawdown - only for period-based formats
    if format_type == 'period' or format_type == 'trades':
        if 'Cum. max. drawdown' in df.columns:
            last_row = df.iloc[-1]
            metrics['Max Drawdown'] = last_row.get('Cum. max. drawdown', 0)
            if pd.isna(metrics['Max Drawdown']):
                metrics['Max Drawdown'] = df['Max. drawdown'].min() if 'Max. drawdown' in df.columns else 0
        
        if 'Cum. net profit' in df.columns:
            metrics['Peak Profit'] = df['Cum. net profit'].max()
    else:
        metrics['Max Drawdown'] = 0
        metrics['Peak Profit'] = metrics['Net Profit']
    
    # Risk metrics
    if 'Net profit' in df.columns and format_type != 'day_of_week':
        returns = df['Net profit'].dropna()
        if len(returns) > 1 and returns.std() > 0:
            metrics['Sharpe Ratio'] = (returns.mean() / returns.std() * np.sqrt(252))
            
            # Sortino
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
            metrics['Sortino Ratio'] = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            metrics['Sharpe Ratio'] = 0
            metrics['Sortino Ratio'] = 0
    
    # Recovery Factor
    if metrics.get('Max Drawdown', 0) != 0:
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
    
    # Filter out day_of_week and other non-time-series formats
    valid_dfs = {name: df for name, df in dataframes.items() 
                 if df.attrs.get('format_type', 'period') in ['period', 'trades']}
    
    if not valid_dfs:
        st.warning("Portfolio analysis requires time-series data (Daily, Weekly, Monthly, or Yearly formats)")
        return {}, None
    
    # Default to equal weights if not specified
    if weights is None:
        weights = {name: 1.0 for name in valid_dfs.keys()}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Combine data by Period
    combined_data = []
    
    for name, df in valid_dfs.items():
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
    
    combined = combined.sort_values('Period').ffill().fillna(0)
    
    # Calculate combined cumulative profit
    cum_cols = [col for col in combined.columns if '_Cum' in col]
    combined['Portfolio_Cum'] = combined[cum_cols].sum(axis=1)
    
    # Calculate combined net profit per period
    net_cols = [col for col in combined.columns if '_Net' in col]
    combined['Portfolio_Net'] = combined[net_cols].sum(axis=1)
    
    # Calculate portfolio metrics
    portfolio_metrics = {}
    portfolio_metrics['Net Profit'] = combined['Portfolio_Cum'].iloc[-1]
    portfolio_metrics['Avg Period Return'] = combined['Portfolio_Net'].mean()
    portfolio_metrics['Std Dev'] = combined['Portfolio_Net'].std()
    
    if portfolio_metrics['Std Dev'] > 0:
        portfolio_metrics['Sharpe Ratio'] = (portfolio_metrics['Avg Period Return'] / 
                                            portfolio_metrics['Std Dev'] * np.sqrt(252))
    else:
        portfolio_metrics['Sharpe Ratio'] = 0
    
    # Max drawdown
    running_max = combined['Portfolio_Cum'].cummax()
    drawdown = combined['Portfolio_Cum'] - running_max
    portfolio_metrics['Max Drawdown'] = drawdown.min()
    portfolio_metrics['Peak Profit'] = running_max.max()
    
    if portfolio_metrics['Max Drawdown'] != 0:
        portfolio_metrics['Recovery Factor'] = abs(portfolio_metrics['Net Profit'] / 
                                                   portfolio_metrics['Max Drawdown'])
    else:
        portfolio_metrics['Recovery Factor'] = 0
    
    # Winning periods
    winning = len(combined[combined['Portfolio_Net'] > 0])
    portfolio_metrics['Period Win Rate'] = (winning / len(combined) * 100) if len(combined) > 0 else 0
    
    return portfolio_metrics, combined

def format_metric(value, metric_type='currency'):
    """Format metrics for display"""
    if pd.isna(value):
        return "N/A"
    if metric_type == 'currency':
        return f"${value:,.2f}"
    elif metric_type == 'percent':
        return f"{value:.2f}%"
    elif metric_type == 'ratio':
        return f"{value:.2f}"
    else:
        return f"{value:.2f}"

# ==================== MAIN APP ====================

st.markdown('<p class="main-header">📊 NinjaTrader Trading Analytics Dashboard</p>', unsafe_allow_html=True)

# File upload section
st.sidebar.header("📁 Upload Trading Data")
st.sidebar.markdown("Upload one or more NinjaTrader export files (CSV)")
st.sidebar.markdown("**Supported formats:**")
st.sidebar.markdown("- Daily/Weekly/Monthly/Yearly summaries")
st.sidebar.markdown("- Day of Week analysis")
st.sidebar.markdown("- Individual Trade exports")

uploaded_files = st.sidebar.file_uploader(
    "Choose CSV files", 
    type=['csv', 'xlsx'],
    accept_multiple_files=True
)

# Process uploaded files
dataframes = {}
if uploaded_files:
    for file in uploaded_files:
        df = parse_ninjatrader_file(file)
        if df is not None:
            strategy_name = file.name.replace('.csv', '').replace('.xlsx', '')
            dataframes[strategy_name] = df
            format_type = df.attrs.get('format_type', 'unknown')
            st.sidebar.success(f"✅ {strategy_name} ({format_type} format)")

if not dataframes:
    st.info("👆 Please upload one or more NinjaTrader CSV files to begin analysis")
    st.markdown("""
    ### Quick Start Guide
    
    1. **Export data from NinjaTrader** using any of these formats:
       - Performance > Period Summary (Daily, Weekly, Monthly, Yearly)
       - Performance > Day of Week Analysis
       - Trade Performance > Executions
    
    2. **Upload your CSV files** using the sidebar
    
    3. **Explore your analytics** across multiple tabs:
       - Overview metrics and comparisons
       - Time series analysis
       - Performance distributions
       - Portfolio analysis (combine multiple strategies)
       - And more!
    """)
    st.stop()

# Create tabs
tab_names = ["📈 Overview", "📊 Performance", "📉 Risk Metrics", "⏰ Time Analysis", 
             "🎯 Portfolio", "💧 Drawdown", "📅 Period Analysis"]
tabs = st.tabs(tab_names)

# ==================== TAB 1: OVERVIEW ====================
with tabs[0]:
    st.header("Strategy Overview")
    
    if len(dataframes) == 1:
        # Single strategy view
        strategy_name = list(dataframes.keys())[0]
        df = dataframes[strategy_name]
        format_type = df.attrs.get('format_type', 'period')
        
        st.subheader(f"📊 {strategy_name}")
        st.caption(f"Format: {format_type}")
        
        metrics = calculate_summary_metrics(df)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Net Profit", format_metric(metrics.get('Net Profit', 0)))
            st.metric("Win Rate", format_metric(metrics.get('Win Rate', 0), 'percent'))
        
        with col2:
            st.metric("Profit Factor", format_metric(metrics.get('Profit Factor', 0), 'ratio'))
            st.metric("Avg Trade", format_metric(metrics.get('Avg Trade', 0)))
        
        with col3:
            st.metric("Max Drawdown", format_metric(metrics.get('Max Drawdown', 0)))
            st.metric("Recovery Factor", format_metric(metrics.get('Recovery Factor', 0), 'ratio'))
        
        with col4:
            st.metric("Sharpe Ratio", format_metric(metrics.get('Sharpe Ratio', 0), 'ratio'))
            st.metric("Total Periods", f"{metrics.get('Total Periods', 0)}")
        
        # Additional metrics
        st.subheader("Detailed Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💰 Profit Metrics**")
            st.write(f"Gross Profit: {format_metric(metrics.get('Gross Profit', 0))}")
            st.write(f"Gross Loss: {format_metric(metrics.get('Gross Loss', 0))}")
            st.write(f"Commission: {format_metric(metrics.get('Total Commission', 0))}")
            st.write(f"Peak Profit: {format_metric(metrics.get('Peak Profit', 0))}")
        
        with col2:
            st.markdown("**📊 Trade Metrics**")
            st.write(f"Avg Winner: {format_metric(metrics.get('Avg Winner', 0))}")
            st.write(f"Avg Loser: {format_metric(metrics.get('Avg Loser', 0))}")
            st.write(f"Largest Winner: {format_metric(metrics.get('Largest Winner', 0))}")
            st.write(f"Largest Loser: {format_metric(metrics.get('Largest Loser', 0))}")
        
        with col3:
            st.markdown("**🎯 Performance Metrics**")
            st.write(f"Sortino Ratio: {format_metric(metrics.get('Sortino Ratio', 0), 'ratio')}")
            st.write(f"Period Win Rate: {format_metric(metrics.get('Period Win Rate', 0), 'percent')}")
            st.write(f"Profitable Periods: {metrics.get('Profitable Periods', 0)}")
            st.write(f"Avg MAE: {format_metric(metrics.get('Avg MAE', 0))}")
        
        # Equity curve (only for time-series formats)
        if format_type in ['period', 'trades'] and 'Cum. net profit' in df.columns:
            st.subheader("Equity Curve")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            cumulative = df['Cum. net profit'].dropna()
            if format_type == 'trades':
                x_values = range(len(cumulative))
                ax.plot(x_values, cumulative, linewidth=2, color='#2ecc71')
                ax.set_xlabel('Trade Number', fontsize=12)
            else:
                ax.plot(df.index, cumulative, linewidth=2, color='#2ecc71')
                ax.set_xlabel('Period', fontsize=12)
            
            ax.set_title(f'{strategy_name} - Cumulative Profit', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            
            # Add stats box
            stats_text = f"Net: {format_metric(metrics.get('Net Profit', 0))}\n"
            stats_text += f"DD: {format_metric(metrics.get('Max Drawdown', 0))}\n"
            stats_text += f"PF: {format_metric(metrics.get('Profit Factor', 0), 'ratio')}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5), fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        # Multiple strategies comparison
        st.subheader("📊 Strategy Comparison")
        
        # Calculate metrics for all strategies
        all_metrics = {}
        for name, df in dataframes.items():
            all_metrics[name] = calculate_summary_metrics(df)
        
        # Create comparison dataframe
        comparison_metrics = [
            'Net Profit', 'Profit Factor', 'Win Rate', 'Sharpe Ratio', 
            'Max Drawdown', 'Recovery Factor', 'Avg Trade', 'Total Periods'
        ]
        
        comparison_data = []
        for metric in comparison_metrics:
            row = {'Metric': metric}
            for name in all_metrics.keys():
                value = all_metrics[name].get(metric, 0)
                if metric in ['Win Rate', 'Period Win Rate']:
                    row[name] = format_metric(value, 'percent')
                elif metric in ['Profit Factor', 'Sharpe Ratio', 'Sortino Ratio', 'Recovery Factor']:
                    row[name] = format_metric(value, 'ratio')
                elif metric == 'Total Periods':
                    row[name] = int(value)
                else:
                    row[name] = format_metric(value)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.subheader("Visual Comparison")
        
        metrics_to_plot = ['Net Profit', 'Profit Factor', 'Win Rate', 'Sharpe Ratio']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            strategies = list(all_metrics.keys())
            values = [all_metrics[s].get(metric, 0) for s in strategies]
            
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
            ax.bar(strategies, values, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            
            if metric in ['Win Rate']:
                ax.set_ylabel('%', fontsize=10)
            elif metric in ['Net Profit', 'Avg Trade', 'Max Drawdown']:
                ax.set_ylabel('$', fontsize=10)
            
            # Rotate x labels if many strategies
            if len(strategies) > 3:
                ax.set_xticklabels(strategies, rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)

# ==================== TAB 2: PERFORMANCE ====================
with tabs[1]:
    st.header("Performance Analysis")
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="perf_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    
    if 'Net profit' in df.columns:
        st.subheader("Period Performance")
        
        net_profit = df['Net profit'].dropna()
        
        if len(net_profit) > 0:
            # Create performance visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar chart of returns
            if format_type == 'day_of_week':
                x_labels = df['Period']
                x_pos = range(len(x_labels))
                colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in net_profit]
                ax1.bar(x_pos, net_profit, color=colors, alpha=0.7, edgecolor='black')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(x_labels, rotation=0)
                ax1.set_xlabel('Day of Week', fontsize=12)
            else:
                colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in net_profit]
                ax1.bar(range(len(net_profit)), net_profit, color=colors, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Period', fontsize=12)
            
            ax1.set_title('Period Returns', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Net Profit ($)', fontsize=12)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Win/Loss distribution
            wins = net_profit[net_profit > 0]
            losses = net_profit[net_profit < 0]
            
            if len(wins) > 0 or len(losses) > 0:
                categories = []
                values = []
                colors_pie = []
                
                if len(wins) > 0:
                    categories.append(f'Wins ({len(wins)})')
                    values.append(wins.sum())
                    colors_pie.append('#2ecc71')
                
                if len(losses) > 0:
                    categories.append(f'Losses ({len(losses)})')
                    values.append(abs(losses.sum()))
                    colors_pie.append('#e74c3c')
                
                if len(values) > 0:
                    ax2.pie(values, labels=categories, autopct='%1.1f%%', 
                           colors=colors_pie, startangle=90)
                    ax2.set_title('Win/Loss Distribution (by $)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Returns", format_metric(net_profit.sum()))
            with col2:
                st.metric("Mean Return", format_metric(net_profit.mean()))
            with col3:
                st.metric("Best Period", format_metric(net_profit.max()))
            with col4:
                st.metric("Worst Period", format_metric(net_profit.min()))
    
    # Winning % by period type
    if format_type == 'day_of_week' and 'Period' in df.columns and '% Win' in df.columns:
        st.subheader("Win Rate by Day of Week")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        days = df['Period']
        win_rates = df['% Win']
        
        colors = ['#2ecc71' if x >= 50 else '#e74c3c' for x in win_rates]
        ax.bar(days, win_rates, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_title('Win Rate by Day of Week', fontsize=14, fontweight='bold')
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

# ==================== TAB 3: RISK METRICS ====================
with tabs[2]:
    st.header("Risk Analysis")
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="risk_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    metrics = calculate_summary_metrics(df)
    
    # Risk metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📉 Downside Risk")
        st.metric("Max Drawdown", format_metric(metrics.get('Max Drawdown', 0)))
        st.metric("Largest Loss", format_metric(metrics.get('Largest Loser', 0)))
        st.metric("Avg Loss", format_metric(metrics.get('Avg Loser', 0)))
    
    with col2:
        st.subheader("⚖️ Risk-Adjusted Returns")
        st.metric("Sharpe Ratio", format_metric(metrics.get('Sharpe Ratio', 0), 'ratio'))
        st.metric("Sortino Ratio", format_metric(metrics.get('Sortino Ratio', 0), 'ratio'))
        st.metric("Recovery Factor", format_metric(metrics.get('Recovery Factor', 0), 'ratio'))
    
    with col3:
        st.subheader("📊 Efficiency Metrics")
        st.metric("Profit Factor", format_metric(metrics.get('Profit Factor', 0), 'ratio'))
        st.metric("Win Rate", format_metric(metrics.get('Win Rate', 0), 'percent'))
        st.metric("Avg MAE", format_metric(metrics.get('Avg MAE', 0)))
    
    # Risk visualization
    if 'Net profit' in df.columns and format_type != 'day_of_week':
        st.subheader("Return Distribution")
        
        returns = df['Net profit'].dropna()
        
        if len(returns) > 5:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            ax1.hist(returns, bins=min(30, len(returns)//2), alpha=0.7, color='steelblue', edgecolor='black')
            ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${returns.mean():.2f}')
            ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${returns.median():.2f}')
            ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax1.set_title('Return Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Net Profit ($)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot approximation (sorted returns)
            sorted_returns = np.sort(returns)
            theoretical_quantiles = np.linspace(returns.min(), returns.max(), len(sorted_returns))
            
            ax2.scatter(theoretical_quantiles, sorted_returns, alpha=0.5)
            ax2.plot([returns.min(), returns.max()], [returns.min(), returns.max()], 
                    'r--', linewidth=2, label='Normal')
            ax2.set_title('Returns vs Uniform Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Theoretical Values', fontsize=12)
            ax2.set_ylabel('Observed Returns', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Skewness", f"{returns.skew():.2f}")
            with col2:
                st.metric("Kurtosis", f"{returns.kurtosis():.2f}")
            with col3:
                st.metric("Std Dev", format_metric(returns.std()))
            with col4:
                downside = returns[returns < 0]
                st.metric("Downside Dev", format_metric(downside.std() if len(downside) > 0 else 0))

# ==================== TAB 4: TIME ANALYSIS ====================
with tabs[3]:
    st.header("Time-Based Analysis")
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="time_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    
    if format_type == 'day_of_week':
        st.subheader("Performance by Day of Week")
        
        if 'Period' in df.columns and 'Net profit' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            days = df['Period']
            profits = df['Net profit']
            
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in profits]
            ax.bar(days, profits, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_title('Net Profit by Day of Week', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Net Profit ($)', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show detailed table
            st.subheader("Day of Week Statistics")
            display_cols = [col for col in ['Period', '# ', 'Net profit', 'Gross profit', 'Gross loss', 
                                            '% Win', 'Avg. trade'] if col in df.columns]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    
    elif format_type in ['period', 'trades']:
        if 'Month_Year' in df.columns and 'Net profit' in df.columns:
            st.subheader("Monthly Performance")
            
            monthly = df.groupby('Month_Year')['Net profit'].sum().sort_index()
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in monthly.values]
            ax.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_title('Monthly Returns', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Net Profit ($)', fontsize=12)
            ax.set_xticks(range(len(monthly)))
            ax.set_xticklabels(monthly.index, rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Monthly stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Month", format_metric(monthly.max()))
            with col2:
                st.metric("Worst Month", format_metric(monthly.min()))
            with col3:
                winning_months = len(monthly[monthly > 0])
                st.metric("Winning Months", f"{winning_months}/{len(monthly)}")
            with col4:
                st.metric("Avg Monthly", format_metric(monthly.mean()))
        
        # Year analysis if available
        if 'Year' in df.columns and 'Net profit' in df.columns and df['Year'].nunique() > 1:
            st.subheader("Yearly Performance")
            
            yearly = df.groupby('Year')['Net profit'].sum()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in yearly.values]
            ax.bar(yearly.index, yearly.values, color=colors, alpha=0.7, edgecolor='black', width=0.6)
            
            ax.set_title('Yearly Returns', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Net Profit ($)', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        st.info("Time analysis not available for this data format")

# ==================== TAB 5: PORTFOLIO ====================
with tabs[4]:
    st.header("Portfolio Analysis")
    
    if len(dataframes) < 2:
        st.info("Upload multiple strategies to see portfolio analysis")
    else:
        st.subheader("Position Sizing")
        
        # Allow user to set weights
        st.markdown("Set position sizes for each strategy:")
        
        weights = {}
        cols = st.columns(min(len(dataframes), 4))
        
        for idx, name in enumerate(dataframes.keys()):
            with cols[idx % 4]:
                weights[name] = st.number_input(
                    f"{name}", 
                    min_value=0.0, 
                    max_value=10.0, 
                    value=1.0, 
                    step=0.1,
                    key=f"weight_{name}"
                )
        
        # Calculate portfolio metrics
        portfolio_metrics, combined = calculate_portfolio_metrics(dataframes, weights)
        
        if portfolio_metrics and combined is not None:
            # Display portfolio metrics
            st.subheader("Portfolio Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Net Profit", format_metric(portfolio_metrics.get('Net Profit', 0)))
                st.metric("Sharpe Ratio", format_metric(portfolio_metrics.get('Sharpe Ratio', 0), 'ratio'))
            
            with col2:
                st.metric("Max Drawdown", format_metric(portfolio_metrics.get('Max Drawdown', 0)))
                st.metric("Recovery Factor", format_metric(portfolio_metrics.get('Recovery Factor', 0), 'ratio'))
            
            with col3:
                st.metric("Peak Profit", format_metric(portfolio_metrics.get('Peak Profit', 0)))
                st.metric("Avg Period", format_metric(portfolio_metrics.get('Avg Period Return', 0)))
            
            with col4:
                st.metric("Std Dev", format_metric(portfolio_metrics.get('Std Dev', 0)))
                st.metric("Period Win Rate", format_metric(portfolio_metrics.get('Period Win Rate', 0), 'percent'))
            
            # Portfolio equity curve
            st.subheader("Portfolio Equity Curve")
            
            # Add benchmark comparison
            add_benchmark = st.checkbox("Add Benchmark Comparison (SPY)", value=False)
            
            # Prepare monthly data
            monthly_combined = combined.groupby('Month_Year').agg({
                'Portfolio_Cum': 'last'
            }).sort_index()
            
            # Add individual strategies
            for name in dataframes.keys():
                cum_col = f'{name}_Cum'
                if cum_col in combined.columns:
                    strategy_monthly = combined.groupby('Month_Year')[cum_col].last().sort_index()
                    monthly_combined[name] = strategy_monthly
            
            # Fetch benchmark if requested
            if add_benchmark and 'Month_Year' in monthly_combined.index.name or True:
                try:
                    benchmark_capital = st.number_input(
                        "Benchmark Starting Capital ($)", 
                        min_value=1000, 
                        value=10000, 
                        step=1000
                    )
                    
                    # Get date range from portfolio
                    if len(combined) > 0:
                        start_date = combined['Period'].min()
                        end_date = combined['Period'].max()
                        
                        market_data = yf.download('SPY', start=start_date, 
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
                    st.metric(col, format_metric(final_values[col]))

# ==================== TAB 6: DRAWDOWN ANALYSIS ====================
with tabs[5]:
    st.header("Drawdown Analysis")
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="dd_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    
    if format_type not in ['period', 'trades']:
        st.info("Drawdown analysis requires time-series data (Daily, Weekly, Monthly, Yearly, or Trades format)")
    elif 'Cum. net profit' in df.columns:
        cumulative = df['Cum. net profit'].dropna()
        
        if len(cumulative) > 0:
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max
            drawdown_pct = (drawdown / running_max * 100).replace([np.inf, -np.inf], 0).fillna(0)
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Equity curve with drawdown
            ax1.plot(cumulative.index, cumulative.values, label='Equity Curve', linewidth=2, color='#2ecc71')
            ax1.fill_between(cumulative.index, cumulative.values, running_max.values, 
                            alpha=0.3, color='#e74c3c', label='Drawdown')
            ax1.plot(running_max.index, running_max.values, label='Peak Equity', 
                    linewidth=1, linestyle='--', color='#3498db', alpha=0.7)
            ax1.set_title(f'{selected_strategy} - Equity Curve with Drawdown', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Profit ($)', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown chart
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='#e74c3c')
            ax2.plot(drawdown.index, drawdown.values, linewidth=1.5, color='#c0392b')
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
                st.metric("Max Drawdown", format_metric(drawdown.min()))
            with col2:
                max_dd_pct = drawdown_pct.min()
                st.metric("Max DD %", format_metric(max_dd_pct if not np.isnan(max_dd_pct) else 0, 'percent'))
            with col3:
                st.metric("Current Drawdown", format_metric(drawdown.iloc[-1]))
            with col4:
                in_drawdown = drawdown < 0
                current_dd_duration = 0
                for is_dd in reversed(in_drawdown.values):
                    if is_dd:
                        current_dd_duration += 1
                    else:
                        break
                st.metric("Current DD Duration", f"{current_dd_duration} periods")
    else:
        st.info("No cumulative profit data available for drawdown analysis")

# ==================== TAB 7: PERIOD ANALYSIS ====================
with tabs[6]:
    st.header("Period-by-Period Analysis")
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="period_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    
    # Period performance distribution
    if 'Net profit' in df.columns:
        returns = df['Net profit'].dropna()
        
        if len(returns) > 0:
            st.subheader("Period Returns Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            ax1.hist(returns, bins=min(30, max(10, len(returns)//3)), alpha=0.7, color='steelblue', edgecolor='black')
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
                st.metric("Mean Return", format_metric(returns.mean()))
            with col2:
                st.metric("Median Return", format_metric(returns.median()))
            with col3:
                st.metric("Std Dev", format_metric(returns.std()))
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
st.caption("Built for NinjaTrader traders | Powered by Streamlit | Supports all NinjaTrader export formats")
