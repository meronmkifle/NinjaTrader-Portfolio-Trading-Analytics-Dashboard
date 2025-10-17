import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="Trading Analytics Dashboard", layout="wide")

# Helper Functions
def calculate_metrics(df):
    """Calculate various trading metrics"""
    metrics = {}
    
    # Win Rate
    if 'Profit' in df.columns:
        winning_trades = len(df[df['Profit'] > 0])
        total_trades = len(df)
        metrics['Win Rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        metrics['Total Trades'] = total_trades
        metrics['Winning Trades'] = winning_trades
        metrics['Losing Trades'] = total_trades - winning_trades
        
        # Average Win/Loss
        wins = df[df['Profit'] > 0]['Profit']
        losses = df[df['Profit'] < 0]['Profit']
        metrics['Avg Win'] = wins.mean() if len(wins) > 0 else 0
        metrics['Avg Loss'] = losses.mean() if len(losses) > 0 else 0
        metrics['Profit Factor'] = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 0
        
    # Total P&L
    if 'Cum. net profit' in df.columns:
        metrics['Total Profit'] = df['Cum. net profit'].iloc[-1]
        metrics['Peak Profit'] = df['Cum. net profit'].max()
        
    return metrics

def calculate_drawdown(df):
    """Calculate drawdown"""
    if 'Cum. net profit' in df.columns:
        cumulative = df['Cum. net profit']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max)
        drawdown_pct = (drawdown / running_max * 100).fillna(0)
        
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()
        
        return drawdown, drawdown_pct, max_dd, max_dd_pct
    return None, None, 0, 0

def calculate_sortino_ratio(returns, rf_rate=0.0):
    """Calculate Sortino Ratio"""
    excess_returns = returns - rf_rate
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std != 0:
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(252)  # Annualized
        return sortino
    return 0

def calculate_calmar_ratio(returns, max_drawdown):
    """Calculate Calmar Ratio"""
    annual_return = returns.mean() * 252  # Annualized
    if max_drawdown != 0:
        calmar = annual_return / abs(max_drawdown)
        return calmar
    return 0

def calculate_sharpe_ratio(returns, rf_rate=0.0):
    """Calculate Sharpe Ratio"""
    excess_returns = returns - rf_rate
    if excess_returns.std() != 0:
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe
    return 0

def process_trading_data(df, date_col='Entry time', profit_col='Cum. net profit'):
    """Process trading data with date and profit columns"""
    # Parse dates
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y %H:%M', errors='coerce')
        if df[date_col].isna().any():
            df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
    
    # Clean profit column
    if profit_col in df.columns:
        df[profit_col] = df[profit_col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df[profit_col] = df[profit_col].astype(float)
    
    # Extract time features
    if date_col in df.columns:
        df['Month_Year'] = df[date_col].dt.strftime('%Y-%m')
        df['Date'] = df[date_col].dt.date
        df['Day_of_Week'] = df[date_col].dt.day_name()
        df['Hour'] = df[date_col].dt.hour
        df['Week'] = df[date_col].dt.isocalendar().week
        df['Year'] = df[date_col].dt.year
    
    return df

# Streamlit App
st.title("📊 Trading Analytics Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_files = st.file_uploader(
        "Upload Trading CSV Files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files with trading data"
    )
    
    st.markdown("---")
    
    if uploaded_files:
        strategy_names = {}
        for i, file in enumerate(uploaded_files):
            default_name = file.name.replace('.csv', '')
            strategy_names[i] = st.text_input(
                f"Strategy Name {i+1}",
                value=default_name,
                key=f"name_{i}"
            )
    
    st.markdown("---")
    
    # Buy & Hold Options
    add_buy_hold = st.checkbox("Add Buy & Hold Comparison", value=False)
    if add_buy_hold:
        ticker = st.text_input("Ticker Symbol", value="ES=F", help="e.g., ES=F for E-mini S&P 500")
        initial_capital = st.number_input("Initial Capital", value=100000, step=10000)

# Main Content
if uploaded_files:
    # Load and process data
    dataframes = {}
    
    for i, file in enumerate(uploaded_files):
        df = pd.read_csv(file)
        strategy_name = strategy_names.get(i, f"Strategy {i+1}")
        
        # Detect date column
        date_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'period' in col.lower()]
        date_col = date_cols[0] if date_cols else df.columns[0]
        
        # Process data
        df = process_trading_data(df, date_col=date_col)
        dataframes[strategy_name] = df
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Overview",
        "📊 Detailed Metrics",
        "📉 Drawdown Analysis",
        "📅 Trade Distribution",
        "🔍 Risk Analysis"
    ])
    
    # Tab 1: Performance Overview
    with tab1:
        st.header("Performance Overview")
        
        # Create combined monthly profit dataframe
        monthly_data = {}
        for name, df in dataframes.items():
            if 'Month_Year' in df.columns and 'Cum. net profit' in df.columns:
                monthly = df.groupby('Month_Year')['Cum. net profit'].last().sort_index()
                monthly_data[name] = monthly
        
        if monthly_data:
            combined_df = pd.DataFrame(monthly_data)
            combined_df = combined_df.fillna(method='ffill').fillna(0)
            combined_df['Combined'] = combined_df.sum(axis=1)
            
            # Add Buy & Hold if selected
            if add_buy_hold and ticker:
                try:
                    with st.spinner("Fetching market data..."):
                        start_date = combined_df.index.min() + '-01'
                        end_date = combined_df.index.max() + '-28'
                        
                        market_data = yf.download(ticker, start=start_date, end=end_date, interval="1mo", progress=False)
                        market_data['Month_Year'] = market_data.index.strftime('%Y-%m')
                        monthly_prices = market_data.groupby('Month_Year')['Close'].last()
                        
                        first_price = monthly_prices.iloc[0]
                        buy_hold_returns = ((monthly_prices / first_price) - 1) * initial_capital
                        combined_df['Buy & Hold'] = buy_hold_returns.reindex(combined_df.index).fillna(method='ffill').fillna(0)
                except Exception as e:
                    st.error(f"Error fetching market data: {e}")
            
            # Plot
            fig, ax = plt.subplots(figsize=(14, 7))
            
            for col in combined_df.columns:
                if col == 'Combined':
                    ax.plot(combined_df.index, combined_df[col], label=col, linewidth=2, color='black')
                elif col == 'Buy & Hold':
                    ax.plot(combined_df.index, combined_df[col], label=col, linewidth=1.5, linestyle='--', color='red')
                else:
                    ax.plot(combined_df.index, combined_df[col], label=col, linewidth=1)
            
            ax.set_title('Monthly Cumulative Profit Over Time - Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month-Year')
            ax.set_ylabel('Cumulative Profit ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Show every 6th label
            x_ticks = range(0, len(combined_df.index), 6)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([combined_df.index[i] for i in x_ticks], rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Summary metrics in columns
            st.markdown("### Summary Statistics")
            cols = st.columns(len(combined_df.columns))
            
            for i, col in enumerate(combined_df.columns):
                with cols[i]:
                    final_profit = combined_df[col].iloc[-1]
                    peak_profit = combined_df[col].max()
                    
                    st.metric(
                        label=col,
                        value=f"${final_profit:,.2f}",
                        delta=f"Peak: ${peak_profit:,.2f}"
                    )
    
    # Tab 2: Detailed Metrics
    with tab2:
        st.header("Detailed Trading Metrics")
        
        for name, df in dataframes.items():
            st.subheader(f"📌 {name}")
            
            metrics = calculate_metrics(df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", f"{metrics.get('Total Trades', 0):,.0f}")
                st.metric("Win Rate", f"{metrics.get('Win Rate', 0):.2f}%")
            
            with col2:
                st.metric("Winning Trades", f"{metrics.get('Winning Trades', 0):,.0f}")
                st.metric("Losing Trades", f"{metrics.get('Losing Trades', 0):,.0f}")
            
            with col3:
                st.metric("Avg Win", f"${metrics.get('Avg Win', 0):,.2f}")
                st.metric("Avg Loss", f"${metrics.get('Avg Loss', 0):,.2f}")
            
            with col4:
                st.metric("Total Profit", f"${metrics.get('Total Profit', 0):,.2f}")
                st.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")
            
            st.markdown("---")
    
    # Tab 3: Drawdown Analysis
    with tab3:
        st.header("Drawdown Analysis")
        
        for name, df in dataframes.items():
            st.subheader(f"📌 {name}")
            
            drawdown, drawdown_pct, max_dd, max_dd_pct = calculate_drawdown(df)
            
            if drawdown is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Max Drawdown", f"${max_dd:,.2f}")
                    st.metric("Max Drawdown %", f"{max_dd_pct:.2f}%")
                
                # Plot drawdown
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                ax1.plot(df.index, df['Cum. net profit'], label='Cumulative Profit')
                ax1.fill_between(df.index, df['Cum. net profit'], df['Cum. net profit'].cummax(), alpha=0.3, color='red', label='Drawdown')
                ax1.set_title('Cumulative Profit with Drawdown')
                ax1.set_ylabel('Profit ($)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.fill_between(df.index, drawdown, 0, alpha=0.5, color='red')
                ax2.set_title('Drawdown Over Time')
                ax2.set_xlabel('Trade Number')
                ax2.set_ylabel('Drawdown ($)')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("---")
    
    # Tab 4: Trade Distribution
    with tab4:
        st.header("Trade Distribution Analysis")
        
        for name, df in dataframes.items():
            st.subheader(f"📌 {name}")
            
            col1, col2 = st.columns(2)
            
            # Trades by day of week
            with col1:
                if 'Day_of_Week' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_counts = df['Day_of_Week'].value_counts().reindex(day_order, fill_value=0)
                    day_counts.plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title('Trades by Day of Week')
                    ax.set_xlabel('Day')
                    ax.set_ylabel('Number of Trades')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Trades by month
            with col2:
                if 'Month_Year' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    month_counts = df['Month_Year'].value_counts().sort_index()
                    month_counts.plot(kind='bar', ax=ax, color='coral')
                    ax.set_title('Trades by Month')
                    ax.set_xlabel('Month-Year')
                    ax.set_ylabel('Number of Trades')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Trades by hour (if available)
            if 'Hour' in df.columns:
                fig, ax = plt.subplots(figsize=(12, 5))
                hour_counts = df['Hour'].value_counts().sort_index()
                hour_counts.plot(kind='bar', ax=ax, color='green')
                ax.set_title('Trades by Hour of Day')
                ax.set_xlabel('Hour')
                ax.set_ylabel('Number of Trades')
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("---")
    
    # Tab 5: Risk Analysis
    with tab5:
        st.header("Risk-Adjusted Performance Metrics")
        
        for name, df in dataframes.items():
            st.subheader(f"📌 {name}")
            
            if 'Profit' in df.columns:
                returns = df['Profit']
                
                # Calculate ratios
                sharpe = calculate_sharpe_ratio(returns)
                sortino = calculate_sortino_ratio(returns)
                
                drawdown, drawdown_pct, max_dd, max_dd_pct = calculate_drawdown(df)
                calmar = calculate_calmar_ratio(returns, max_dd_pct)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                
                with col2:
                    st.metric("Sortino Ratio", f"{sortino:.3f}")
                
                with col3:
                    st.metric("Calmar Ratio", f"{calmar:.3f}")
                
                # Profit distribution
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histogram
                ax1.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: ${returns.mean():.2f}')
                ax1.set_title('Distribution of Trade Returns')
                ax1.set_xlabel('Profit/Loss ($)')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Box plot
                ax2.boxplot(returns, vert=True)
                ax2.set_title('Trade Returns Box Plot')
                ax2.set_ylabel('Profit/Loss ($)')
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("---")

else:
    st.info("👆 Please upload CSV files in the sidebar to begin analysis")
    
    st.markdown("""
    ### 📋 Expected CSV Format
    
    Your CSV files should contain the following columns:
    - **Date/Time column**: Entry time, Period, Date, etc.
    - **Profit column**: Cum. net profit, Profit, P&L, etc.
    - **Additional columns**: Any other trade data you want to analyze
    
    ### 📊 Features
    
    - **Performance Overview**: Compare multiple strategies and buy & hold
    - **Detailed Metrics**: Win rate, profit factor, average win/loss
    - **Drawdown Analysis**: Visualize and quantify drawdowns
    - **Trade Distribution**: Analyze trades by day, week, month
    - **Risk Analysis**: Sharpe, Sortino, Calmar ratios
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Trading Analytics Dashboard v1.0")
