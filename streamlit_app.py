import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from matplotlib import font_manager
import matplotlib.patches as mpatches

# Page config with custom theme
st.set_page_config(
    page_title="NinjaTrader Analytics Terminal",
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DESIGN SYSTEM ====================

# Brand Colors
COLORS = {
    'navy': '#0A1929',
    'navy_light': '#1A2332',
    'navy_dark': '#050D15',
    'yellow': '#FFD700',
    'yellow_bright': '#FFF44F',
    'yellow_dark': '#D4AF37',
    'black': '#000000',
    'white': '#FFFFFF',
    'gray': '#A0AEC0',
    'gray_light': '#E2E8F0',
    'gray_dark': '#2D3748',
    'green': '#10B981',
    'green_glow': '#059669',
    'red': '#EF4444',
    'red_glow': '#DC2626',
    'blue': '#3B82F6',
    'blue_glow': '#2563EB'
}

# Custom CSS with modern styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['navy_dark']} 0%, {COLORS['navy']} 50%, {COLORS['navy_light']} 100%);
        background-attachment: fixed;
    }}
    
    /* Main container */
    .main {{
        background: transparent;
    }}
    
    /* Headers */
    .main-header {{
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, {COLORS['yellow']} 0%, {COLORS['yellow_bright']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }}
    
    @keyframes glow {{
        from {{
            text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        }}
        to {{
            text-shadow: 0 0 40px rgba(255, 215, 0, 0.5), 0 0 60px rgba(255, 215, 0, 0.3);
        }}
    }}
    
    .sub-header {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: {COLORS['yellow']};
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-left: 4px solid {COLORS['yellow']};
        padding-left: 1rem;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['navy_light']} 0%, {COLORS['navy']} 100%);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 
                    0 0 20px rgba(255, 215, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, {COLORS['yellow']}, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(255, 215, 0, 0.5);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6), 
                    0 0 40px rgba(255, 215, 0, 0.2);
    }}
    
    .metric-card:hover::before {{
        opacity: 1;
    }}
    
    .metric-label {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: {COLORS['gray']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['yellow']};
        line-height: 1;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
    }}
    
    .metric-value.positive {{
        color: {COLORS['green']};
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }}
    
    .metric-value.negative {{
        color: {COLORS['red']};
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.3);
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['navy_dark']} 0%, {COLORS['navy']} 100%);
        border-right: 2px solid rgba(255, 215, 0, 0.2);
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {COLORS['gray_light']};
        font-family: 'Rajdhani', sans-serif;
    }}
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {COLORS['yellow']};
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {COLORS['navy_light']};
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 215, 0, 0.2);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: {COLORS['gray']};
        background: transparent;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255, 215, 0, 0.1);
        color: {COLORS['yellow']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['yellow_dark']} 0%, {COLORS['yellow']} 100%) !important;
        color: {COLORS['navy_dark']} !important;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
    }}
    
    /* Buttons */
    .stButton > button {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, {COLORS['yellow_dark']} 0%, {COLORS['yellow']} 100%);
        color: {COLORS['navy_dark']};
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.5);
        background: linear-gradient(135deg, {COLORS['yellow']} 0%, {COLORS['yellow_bright']} 100%);
    }}
    
    /* Dataframe styling */
    .dataframe {{
        font-family: 'Rajdhani', sans-serif !important;
        background: {COLORS['navy_light']} !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 8px !important;
    }}
    
    /* Success/Info messages */
    .stSuccess, .stInfo {{
        background: linear-gradient(135deg, {COLORS['navy_light']} 0%, {COLORS['navy']} 100%);
        border-left: 4px solid {COLORS['yellow']};
        color: {COLORS['gray_light']};
        font-family: 'Rajdhani', sans-serif;
        border-radius: 8px;
        padding: 1rem;
    }}
    
    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: {COLORS['navy_light']};
        border: 2px dashed rgba(255, 215, 0, 0.3);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: rgba(255, 215, 0, 0.6);
        background: rgba(255, 215, 0, 0.05);
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: {COLORS['yellow']} !important;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 12px;
        height: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['navy_dark']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {COLORS['yellow_dark']} 0%, {COLORS['yellow']} 100%);
        border-radius: 6px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['yellow_bright']};
    }}
    
    /* Chart container */
    .chart-container {{
        background: {COLORS['navy_light']};
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}
</style>
""", unsafe_allow_html=True)

# Set matplotlib style for consistent chart styling
plt.style.use('dark_background')
sns.set_palette([COLORS['yellow'], COLORS['green'], COLORS['blue'], COLORS['red']])

def setup_plot_style(fig, ax):
    """Apply consistent styling to matplotlib plots"""
    fig.patch.set_facecolor(COLORS['navy_light'])
    ax.set_facecolor(COLORS['navy'])
    ax.spines['top'].set_color(COLORS['gray_dark'])
    ax.spines['right'].set_color(COLORS['gray_dark'])
    ax.spines['bottom'].set_color(COLORS['yellow'])
    ax.spines['left'].set_color(COLORS['yellow'])
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(colors=COLORS['gray_light'], which='both', labelsize=10)
    ax.xaxis.label.set_color(COLORS['yellow'])
    ax.yaxis.label.set_color(COLORS['yellow'])
    ax.title.set_color(COLORS['yellow'])
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color=COLORS['yellow'])

# ==================== HELPER FUNCTIONS ====================

def detect_format_type(df):
    """Detect the type of NinjaTrader export format"""
    if 'Period' not in df.columns:
        return 'unknown'
    
    first_period = df['Period'].dropna().iloc[0] if len(df['Period'].dropna()) > 0 else ''
    
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if any(day in str(first_period) for day in days_of_week):
        return 'day_of_week'
    
    if ':' in str(first_period) and ('AM' in str(first_period) or 'PM' in str(first_period)):
        return 'trades'
    
    if 'Cum. net profit' in df.columns:
        has_na = df['Cum. net profit'].astype(str).str.contains('n/a', case=False).any()
        if has_na:
            return 'day_of_week'
    
    return 'period'

def clean_currency_column(series):
    """Clean currency columns (remove $, commas, handle n/a)"""
    cleaned = series.astype(str).str.replace('$', '').str.replace(',', '').str.replace('%', '')
    cleaned = cleaned.replace(['n/a', 'N/A', 'na', 'NA'], np.nan)
    return pd.to_numeric(cleaned, errors='coerce')

def parse_period_column(df, format_type):
    """Parse Period column based on format type"""
    if format_type == 'day_of_week':
        day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df['Day_Order'] = df['Period'].map(day_order)
        df['Period_Original'] = df['Period']
        return df
    
    elif format_type == 'trades':
        try:
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
    
    else:
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
        
        format_type = detect_format_type(df)
        df = parse_period_column(df, format_type)
        
        numeric_cols = ['Cum. net profit', 'Net profit', 'Gross profit', 'Gross loss', 
                       'Commission', 'Cum. max. drawdown', 'Max. drawdown', 
                       'Avg. trade', 'Avg. winner', 'Avg. loser', 
                       'Lrg. winner', 'Lrg. loser', 'Avg. MAE', 'Avg. MFE', 'Avg. ETD']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = clean_currency_column(df[col])
        
        if '% Win' in df.columns:
            df['% Win'] = clean_currency_column(df['% Win'])
        
        if '% Trade' in df.columns:
            df['% Trade'] = clean_currency_column(df['% Trade'])
        
        # Sort by Period chronologically for correct equity curve
        if format_type in ['period', 'trades'] and 'Period' in df.columns:
            df = df.sort_values('Period').reset_index(drop=True)
        elif format_type == 'day_of_week' and 'Day_Order' in df.columns:
            df = df.sort_values('Day_Order').reset_index(drop=True)
        
        df.attrs['format_type'] = format_type
        
        return df
    
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def calculate_summary_metrics(df):
    """Calculate overall metrics from period data"""
    metrics = {}
    format_type = df.attrs.get('format_type', 'period')
    
    if len(df) == 0:
        return metrics
    
    if format_type == 'day_of_week':
        metrics['Net Profit'] = df['Net profit'].sum() if 'Net profit' in df.columns else 0
        metrics['Gross Profit'] = df['Gross profit'].sum() if 'Gross profit' in df.columns else 0
        metrics['Gross Loss'] = abs(df['Gross loss'].sum()) if 'Gross loss' in df.columns else 0
        metrics['Total Commission'] = df['Commission'].sum() if 'Commission' in df.columns else 0
    else:
        last_row = df.iloc[-1]
        metrics['Net Profit'] = last_row.get('Cum. net profit', 0)
        if pd.isna(metrics['Net Profit']):
            metrics['Net Profit'] = df['Net profit'].sum() if 'Net profit' in df.columns else 0
        
        metrics['Gross Profit'] = df['Gross profit'].sum() if 'Gross profit' in df.columns else 0
        metrics['Gross Loss'] = abs(df['Gross loss'].sum()) if 'Gross loss' in df.columns else 0
        metrics['Total Commission'] = df['Commission'].sum() if 'Commission' in df.columns else 0
    
    if metrics['Gross Loss'] != 0:
        metrics['Profit Factor'] = metrics['Gross Profit'] / metrics['Gross Loss']
    else:
        metrics['Profit Factor'] = 0
    
    metrics['Avg Trade'] = df['Avg. trade'].mean() if 'Avg. trade' in df.columns else 0
    metrics['Avg Winner'] = df['Avg. winner'].mean() if 'Avg. winner' in df.columns else 0
    metrics['Avg Loser'] = df['Avg. loser'].mean() if 'Avg. loser' in df.columns else 0
    metrics['Win Rate'] = df['% Win'].mean() if '% Win' in df.columns else 0
    metrics['Largest Winner'] = df['Lrg. winner'].max() if 'Lrg. winner' in df.columns else 0
    metrics['Largest Loser'] = df['Lrg. loser'].min() if 'Lrg. loser' in df.columns else 0
    
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
    
    if 'Net profit' in df.columns and format_type != 'day_of_week':
        returns = df['Net profit'].dropna()
        if len(returns) > 1 and returns.std() > 0:
            metrics['Sharpe Ratio'] = (returns.mean() / returns.std() * np.sqrt(252))
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
            metrics['Sortino Ratio'] = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            metrics['Sharpe Ratio'] = 0
            metrics['Sortino Ratio'] = 0
    
    if metrics.get('Max Drawdown', 0) != 0:
        metrics['Recovery Factor'] = abs(metrics['Net Profit'] / metrics['Max Drawdown'])
    else:
        metrics['Recovery Factor'] = 0
    
    metrics['Avg MAE'] = df['Avg. MAE'].mean() if 'Avg. MAE' in df.columns else 0
    metrics['Avg MFE'] = df['Avg. MFE'].mean() if 'Avg. MFE' in df.columns else 0
    metrics['Total Periods'] = len(df)
    
    if 'Net profit' in df.columns:
        profitable_periods = len(df[df['Net profit'] > 0])
        metrics['Profitable Periods'] = profitable_periods
        metrics['Period Win Rate'] = (profitable_periods / len(df) * 100) if len(df) > 0 else 0
    
    return metrics

def calculate_monthly_returns(dataframes, weights=None, initial_capital=100000):
    """
    Calculate monthly returns for single strategy or portfolio.
    Returns a DataFrame with Year, Month, and Return% columns.
    """
    valid_dfs = {name: df for name, df in dataframes.items() 
                 if df.attrs.get('format_type', 'period') in ['period', 'trades']}
    
    if not valid_dfs:
        return None
    
    # Initialize weights
    if weights is None:
        weights = {name: 1.0 for name in valid_dfs.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Calculate capital allocation
    capital_allocation = {name: initial_capital * weight for name, weight in weights.items()}
    
    # Build monthly data
    monthly_data = {}
    
    for name, df in valid_dfs.items():
        if 'Period' in df.columns and 'Net profit' in df.columns:
            df_copy = df.copy()
            df_copy['Year'] = pd.to_datetime(df_copy['Period']).dt.year
            df_copy['Month'] = pd.to_datetime(df_copy['Period']).dt.month
            
            # Group by year and month
            monthly_grouped = df_copy.groupby(['Year', 'Month'])['Net profit'].sum()
            
            for (year, month), net_profit in monthly_grouped.items():
                key = (year, month)
                if key not in monthly_data:
                    monthly_data[key] = 0
                
                # Add weighted return
                monthly_data[key] += net_profit * weights[name]
    
    # Convert to DataFrame
    monthly_df = pd.DataFrame([
        {
            'Year': year,
            'Month': month,
            'Net_Return': ret
        }
        for (year, month), ret in monthly_data.items()
    ])
    
    # Calculate return percentage based on initial capital
    monthly_df['Return_%'] = (monthly_df['Net_Return'] / initial_capital) * 100
    
    # Sort by year and month
    monthly_df = monthly_df.sort_values(['Year', 'Month'])
    
    return monthly_df

def create_monthly_performance_chart(monthly_df, selected_year=None):
    """
    Create a monthly performance chart matching the image style.
    """
    if monthly_df is None or len(monthly_df) == 0:
        return None
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Top: Bar chart showing selected year monthly returns
    ax_bars = fig.add_subplot(gs[0])
    setup_plot_style(fig, ax_bars)
    
    # Get selected year data (default to latest year if not specified)
    if selected_year is None:
        selected_year = monthly_df['Year'].max()
    year_data = monthly_df[monthly_df['Year'] == selected_year].copy()
    
    # Ensure all 12 months are present
    all_months = pd.DataFrame({'Month': range(1, 13)})
    year_data = all_months.merge(year_data, on='Month', how='left')
    year_data['Return_%'].fillna(0, inplace=True)
    
    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create bars
    colors = [COLORS['green'] if x > 0 else COLORS['red'] if x < 0 else COLORS['gray'] 
              for x in year_data['Return_%']]
    
    bars = ax_bars.bar(range(12), year_data['Return_%'], color=colors, alpha=0.8, 
                       edgecolor=COLORS['yellow'], linewidth=1.5, width=0.7)
    
    # Add glow effect
    for i, bar in enumerate(bars):
        if year_data['Return_%'].iloc[i] != 0:
            ax_bars.bar(i, bar.get_height(), bar.get_width(), 
                       color=bar.get_facecolor(), alpha=0.3, edgecolor='none')
    
    # Formatting
    ax_bars.set_xticks(range(12))
    ax_bars.set_xticklabels(month_labels, fontsize=11, fontweight='600')
    ax_bars.set_title(f'PERFORMANCE - {int(selected_year)}', fontsize=16, fontweight='bold', 
                     family='Rajdhani', pad=20)
    ax_bars.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
    ax_bars.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Add percentage labels on bars
    for i, v in enumerate(year_data['Return_%']):
        if v != 0:
            label_y = v + (1 if v > 0 else -1)
            ax_bars.text(i, label_y, f'{v:.2f}%', ha='center', va='bottom' if v > 0 else 'top',
                        color=COLORS['yellow'], fontsize=9, fontweight='bold')
    
    # Calculate YTD return using compounding (shown in top right)
    # Formula: (1 + r1) × (1 + r2) × ... × (1 + rn) - 1
    # This correctly compounds monthly returns instead of simply summing them
    monthly_returns_decimal = year_data['Return_%'] / 100
    ytd_return = (np.prod(1 + monthly_returns_decimal) - 1) * 100
    ax_bars.text(0.98, 0.95, f'{ytd_return:.2f}%', transform=ax_bars.transAxes,
                ha='right', va='top', fontsize=14, fontweight='bold',
                color=COLORS['green'] if ytd_return > 0 else COLORS['red'],
                bbox=dict(boxstyle='round', facecolor=COLORS['navy'], 
                         edgecolor=COLORS['yellow'], linewidth=2, alpha=0.9))
    
    # Bottom: Heatmap table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    
    # Prepare data for table
    pivot_data = monthly_df.pivot(index='Year', columns='Month', values='Return_%')
    pivot_data = pivot_data.reindex(columns=range(1, 13))
    pivot_data.fillna(0, inplace=True)
    
    # Sort years in descending order
    pivot_data = pivot_data.sort_index(ascending=False)
    
    # Create table
    cell_height = 0.08
    cell_width = 1.0 / 13
    
    for i, year in enumerate(pivot_data.index):
        # Highlight selected year
        is_selected = (year == selected_year)
        year_color = COLORS['yellow_bright'] if is_selected else COLORS['yellow']
        year_weight = 'bold' if is_selected else 'bold'
        year_size = 13 if is_selected else 12
        
        # Year label with highlighting
        if is_selected:
            # Add highlight box for selected year
            highlight_rect = mpatches.Rectangle((0.0, 0.85 - i * cell_height - 0.005), 
                                               0.14, cell_height * 0.95,
                                               facecolor=COLORS['yellow'], alpha=0.15,
                                               edgecolor=COLORS['yellow'], linewidth=2,
                                               transform=ax_table.transAxes, zorder=0)
            ax_table.add_patch(highlight_rect)
        
        ax_table.text(0.02, 0.9 - i * cell_height, str(int(year)), 
                     ha='left', va='center', fontsize=year_size, fontweight=year_weight,
                     color=year_color, family='Rajdhani')
        
        for j, month in enumerate(range(1, 13)):
            value = pivot_data.loc[year, month]
            
            # Determine cell color
            if value > 0:
                cell_color = COLORS['green']
                text_alpha = min(1.0, abs(value) / 10 + 0.3)
            elif value < 0:
                cell_color = COLORS['red']
                text_alpha = min(1.0, abs(value) / 10 + 0.3)
            else:
                cell_color = COLORS['gray_dark']
                text_alpha = 0.3
            
            # Draw cell background with extra border for selected year
            edge_width = 1.5 if is_selected else 0.5
            edge_color = COLORS['yellow'] if is_selected else COLORS['gray_dark']
            
            rect = mpatches.Rectangle((0.15 + j * cell_width, 0.85 - i * cell_height), 
                                     cell_width * 0.95, cell_height * 0.9,
                                     facecolor=cell_color, alpha=text_alpha * 0.3,
                                     edgecolor=edge_color, linewidth=edge_width)
            ax_table.add_patch(rect)
            
            # Add text
            if value != 0:
                ax_table.text(0.15 + j * cell_width + cell_width * 0.475, 
                            0.85 - i * cell_height + cell_height * 0.45,
                            f'{value:.2f}%', ha='center', va='center',
                            fontsize=9, fontweight='600',
                            color=cell_color if value != 0 else COLORS['gray'],
                            family='Rajdhani')
    
    # Add month headers
    for j, month_name in enumerate(month_labels):
        ax_table.text(0.15 + j * cell_width + cell_width * 0.475, 0.95,
                     month_name, ha='center', va='center',
                     fontsize=10, fontweight='bold',
                     color=COLORS['gray_light'], family='Rajdhani')
    
    # Add annual totals column
    ax_table.text(0.15 + 12 * cell_width + cell_width * 0.475, 0.95,
                 'Year', ha='center', va='center',
                 fontsize=10, fontweight='bold',
                 color=COLORS['gray_light'], family='Rajdhani')
    
    for i, year in enumerate(pivot_data.index):
        # Calculate compounded annual return
        monthly_returns_decimal = pivot_data.loc[year] / 100
        annual_return = (np.prod(1 + monthly_returns_decimal.fillna(0)) - 1) * 100
        color = COLORS['green'] if annual_return > 0 else COLORS['red'] if annual_return < 0 else COLORS['gray']
        
        is_selected = (year == selected_year)
        edge_width = 1.5 if is_selected else 1
        edge_color = COLORS['yellow'] if is_selected else COLORS['yellow']
        
        rect = mpatches.Rectangle((0.15 + 12 * cell_width, 0.85 - i * cell_height), 
                                 cell_width * 0.95, cell_height * 0.9,
                                 facecolor=color, alpha=0.2,
                                 edgecolor=edge_color, linewidth=edge_width)
        ax_table.add_patch(rect)
        
        ax_table.text(0.15 + 12 * cell_width + cell_width * 0.475,
                     0.85 - i * cell_height + cell_height * 0.45,
                     f'{annual_return:.2f}%', ha='center', va='center',
                     fontsize=10, fontweight='bold',
                     color=color, family='Rajdhani')
    
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    
    fig.patch.set_facecolor(COLORS['navy_light'])
    plt.tight_layout()
    
    return fig

def calculate_portfolio_metrics(dataframes, weights=None, initial_capital=100000):
    """
    Calculate combined portfolio metrics with proper capital allocation.
    """
    if not dataframes:
        return {}, None
    
    valid_dfs = {name: df for name, df in dataframes.items() 
                 if df.attrs.get('format_type', 'period') in ['period', 'trades']}
    
    if not valid_dfs:
        st.warning("Portfolio analysis requires time-series data (Daily, Weekly, Monthly, or Yearly formats)")
        return {}, None
    
    if weights is None:
        weights = {name: 1.0 for name in valid_dfs.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Calculate capital allocation
    capital_allocation = {name: initial_capital * weight for name, weight in weights.items()}
    
    # Get all unique periods across strategies
    all_periods = set()
    for df in valid_dfs.values():
        if 'Period' in df.columns:
            all_periods.update(df['Period'].dropna())
    
    all_periods = sorted(list(all_periods))
    
    # Build portfolio returns
    portfolio_data = []
    
    for period in all_periods:
        period_return = 0
        period_date = period
        month_year = pd.to_datetime(period).strftime('%Y-%m')
        
        for name, df in valid_dfs.items():
            if 'Period' in df.columns and 'Net profit' in df.columns:
                period_df = df[df['Period'] == period]
                if not period_df.empty:
                    strategy_return = period_df['Net profit'].iloc[0]
                    weighted_return = strategy_return * weights[name]
                    period_return += weighted_return
        
        portfolio_data.append({
            'Period': period_date,
            'Month_Year': month_year,
            'Net_Return': period_return
        })
    
    combined = pd.DataFrame(portfolio_data)
    combined = combined.sort_values('Period')
    
    # Calculate cumulative returns
    combined['Portfolio_Cum'] = initial_capital + combined['Net_Return'].cumsum()
    combined['Portfolio_Net'] = combined['Net_Return']
    
    # Also add individual strategy curves for comparison
    for name, df in valid_dfs.items():
        if 'Period' in df.columns and 'Cum. net profit' in df.columns:
            strategy_cum = df.set_index('Period')['Cum. net profit']
            combined[f'{name}_Cum'] = combined['Period'].map(
                lambda p: strategy_cum.get(p, np.nan) * weights[name]
            ).ffill().fillna(0) + capital_allocation[name]
    
    # Calculate metrics
    portfolio_metrics = {}
    portfolio_metrics['Initial Capital'] = initial_capital
    portfolio_metrics['Net Profit'] = combined['Portfolio_Cum'].iloc[-1] - initial_capital
    portfolio_metrics['Total Return %'] = (portfolio_metrics['Net Profit'] / initial_capital) * 100
    portfolio_metrics['Avg Period Return'] = combined['Portfolio_Net'].mean()
    portfolio_metrics['Std Dev'] = combined['Portfolio_Net'].std()
    
    if portfolio_metrics['Std Dev'] > 0:
        portfolio_metrics['Sharpe Ratio'] = (portfolio_metrics['Avg Period Return'] / 
                                            portfolio_metrics['Std Dev'] * np.sqrt(252))
    else:
        portfolio_metrics['Sharpe Ratio'] = 0
    
    # Max drawdown from equity curve
    running_max = combined['Portfolio_Cum'].cummax()
    drawdown = combined['Portfolio_Cum'] - running_max
    portfolio_metrics['Max Drawdown'] = drawdown.min()
    portfolio_metrics['Max Drawdown %'] = (drawdown.min() / running_max.max() * 100) if running_max.max() > 0 else 0
    portfolio_metrics['Peak Equity'] = running_max.max()
    
    if portfolio_metrics['Max Drawdown'] != 0:
        portfolio_metrics['Recovery Factor'] = abs(portfolio_metrics['Net Profit'] / 
                                                   portfolio_metrics['Max Drawdown'])
    else:
        portfolio_metrics['Recovery Factor'] = 0
    
    winning = len(combined[combined['Portfolio_Net'] > 0])
    portfolio_metrics['Period Win Rate'] = (winning / len(combined) * 100) if len(combined) > 0 else 0
    
    # Calculate portfolio profit factor
    gross_profit = combined[combined['Portfolio_Net'] > 0]['Portfolio_Net'].sum()
    gross_loss = abs(combined[combined['Portfolio_Net'] < 0]['Portfolio_Net'].sum())
    portfolio_metrics['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
    
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

def create_metric_card(label, value, value_type='currency', delta=None):
    """Create a styled metric card with optional comparison"""
    if value_type == 'currency':
        display_value = format_metric(value, 'currency')
        value_class = 'positive' if value > 0 else 'negative' if value < 0 else ''
    elif value_type == 'percent':
        display_value = format_metric(value, 'percent')
        value_class = 'positive' if value > 0 else 'negative' if value < 0 else ''
    elif value_type == 'ratio':
        display_value = format_metric(value, 'ratio')
        value_class = 'positive' if value > 1 else 'negative' if value < 1 else ''
    else:
        display_value = str(value)
        value_class = ''
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {value_class}">{display_value}</div>
        </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================

st.markdown('<p class="main-header">⚡ NinjaTrader Analytics Terminal</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 📁 DATA UPLOAD")
st.sidebar.markdown("Upload NinjaTrader export files (CSV)")

uploaded_files = st.sidebar.file_uploader(
    "Select Files", 
    type=['csv', 'xlsx'],
    accept_multiple_files=True,
    help="Upload one or more trading strategy exports"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### BENCHMARK")
show_benchmark = st.sidebar.checkbox("Add Buy & Hold Comparison", value=False)
benchmark_ticker = None
benchmark_capital = 100000
if show_benchmark:
    benchmark_ticker = st.sidebar.text_input("Ticker", value="ES=F", help="e.g., ES=F, NQ=F, SPY")
    benchmark_capital = st.sidebar.number_input("Capital ($)", value=100000, step=10000)

# Process files
dataframes = {}
if uploaded_files:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### LOADED STRATEGIES")
    for file in uploaded_files:
        df = parse_ninjatrader_file(file)
        if df is not None:
            strategy_name = file.name.replace('.csv', '').replace('.xlsx', '')
            dataframes[strategy_name] = df
            format_type = df.attrs.get('format_type', 'unknown')
            st.sidebar.success(f"✓ {strategy_name} ({format_type})")

if not dataframes:
    st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h2 style='color: #FFD700; font-family: Rajdhani, sans-serif; font-size: 2rem; margin-bottom: 2rem;'>
                WELCOME TO THE ANALYTICS TERMINAL
            </h2>
            <p style='color: #A0AEC0; font-family: Rajdhani, sans-serif; font-size: 1.2rem; line-height: 1.8;'>
                Upload your NinjaTrader strategy exports to begin comprehensive performance analysis.<br><br>
                <strong style='color: #FFD700;'>Supported Formats:</strong><br>
                • Daily/Weekly/Monthly/Yearly Performance Summaries<br>
                • Day of Week Analysis<br>
                • Individual Trade Executions<br><br>
                <strong style='color: #FFD700;'>Features:</strong><br>
                • Multi-strategy portfolio analysis<br>
                • Advanced risk metrics<br>
                • Professional visualizations<br>
                • Benchmark comparisons<br>
                • Monthly performance heatmaps
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Create tabs
tabs = st.tabs(["OVERVIEW", "MONTHLY", "PERFORMANCE", "RISK", "TIME", "PORTFOLIO", "DRAWDOWN", "PERIODS"])

# ==================== TAB 1: OVERVIEW ====================
with tabs[0]:
    st.markdown('<p class="sub-header">Strategy Overview</p>', unsafe_allow_html=True)
    
    if len(dataframes) == 1:
        strategy_name = list(dataframes.keys())[0]
        df = dataframes[strategy_name]
        format_type = df.attrs.get('format_type', 'period')
        metrics = calculate_summary_metrics(df)
        
        st.markdown(f"**{strategy_name}** • *{format_type} format*")
        
        # Key metrics in cards
        cols = st.columns(4)
        with cols[0]:
            create_metric_card("Net Profit", metrics.get('Net Profit', 0), 'currency')
        with cols[1]:
            create_metric_card("Profit Factor", metrics.get('Profit Factor', 0), 'ratio')
        with cols[2]:
            create_metric_card("Win Rate", metrics.get('Win Rate', 0), 'percent')
        with cols[3]:
            create_metric_card("Sharpe Ratio", metrics.get('Sharpe Ratio', 0), 'ratio')
        
        cols = st.columns(4)
        with cols[0]:
            create_metric_card("Max Drawdown", metrics.get('Max Drawdown', 0), 'currency')
        with cols[1]:
            create_metric_card("Recovery Factor", metrics.get('Recovery Factor', 0), 'ratio')
        with cols[2]:
            create_metric_card("Avg Trade", metrics.get('Avg Trade', 0), 'currency')
        with cols[3]:
            create_metric_card("Total Periods", metrics.get('Total Periods', 0), 'other')
        
        # Equity curve
        if format_type in ['period', 'trades'] and 'Cum. net profit' in df.columns:
            st.markdown('<p class="sub-header">Equity Curve</p>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(16, 7))
            setup_plot_style(fig, ax)
            
            cumulative = df['Cum. net profit'].dropna()
            x_values = range(len(cumulative))
            
            ax.plot(x_values, cumulative.values, linewidth=3, color=COLORS['yellow'], 
                   label='Cumulative Profit', zorder=3)
            ax.fill_between(x_values, 0, cumulative.values, alpha=0.2, color=COLORS['yellow'])
            
            if format_type == 'trades':
                ax.set_xlabel('Trade Number', fontsize=14, fontweight='bold')
            else:
                ax.set_xlabel('Period', fontsize=14, fontweight='bold')
                if 'Period' in df.columns:
                    n_ticks = min(10, len(df))
                    tick_indices = np.linspace(0, len(df)-1, n_ticks, dtype=int)
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([df.iloc[i]['Period'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['Period'], 'strftime') else str(df.iloc[i]['Period']) for i in tick_indices], rotation=45, ha='right')
            
            ax.set_title(f'{strategy_name.upper()} - CUMULATIVE PROFIT', 
                        fontsize=18, fontweight='bold', pad=20, family='Rajdhani')
            ax.set_ylabel('Profit ($)', fontsize=14, fontweight='bold')
            ax.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5, zorder=1)
            ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
            
            ax.plot(x_values, cumulative.values, 
                   linewidth=6, color=COLORS['yellow'], alpha=0.2, zorder=2)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Additional metrics
        st.markdown('<p class="sub-header">Detailed Statistics</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**PROFIT METRICS**")
            st.write(f"Gross Profit: {format_metric(metrics.get('Gross Profit', 0))}")
            st.write(f"Gross Loss: {format_metric(metrics.get('Gross Loss', 0))}")
            st.write(f"Commission: {format_metric(metrics.get('Total Commission', 0))}")
            st.write(f"Peak Profit: {format_metric(metrics.get('Peak Profit', 0))}")
        
        with col2:
            st.markdown(f"**TRADE METRICS**")
            st.write(f"Avg Winner: {format_metric(metrics.get('Avg Winner', 0))}")
            st.write(f"Avg Loser: {format_metric(metrics.get('Avg Loser', 0))}")
            st.write(f"Largest Winner: {format_metric(metrics.get('Largest Winner', 0))}")
            st.write(f"Largest Loser: {format_metric(metrics.get('Largest Loser', 0))}")
        
        with col3:
            st.markdown(f"**PERFORMANCE**")
            st.write(f"Sortino Ratio: {format_metric(metrics.get('Sortino Ratio', 0), 'ratio')}")
            st.write(f"Period Win Rate: {format_metric(metrics.get('Period Win Rate', 0), 'percent')}")
            st.write(f"Profitable Periods: {metrics.get('Profitable Periods', 0)}")
            st.write(f"Avg MAE: {format_metric(metrics.get('Avg MAE', 0))}")
    
    else:
        # Multiple strategies comparison
        all_metrics = {}
        for name, df in dataframes.items():
            all_metrics[name] = calculate_summary_metrics(df)
        
        # Comparison charts
        st.markdown('<p class="sub-header">Strategy Comparison</p>', unsafe_allow_html=True)
        
        metrics_to_plot = [
            ('Net Profit', 'currency'),
            ('Profit Factor', 'ratio'),
            ('Win Rate', 'percent'),
            ('Sharpe Ratio', 'ratio')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, (metric, mtype) in enumerate(metrics_to_plot):
            ax = axes[idx]
            setup_plot_style(fig, ax)
            
            strategies = list(all_metrics.keys())
            values = [all_metrics[s].get(metric, 0) for s in strategies]
            
            colors = [COLORS['green'] if v > 0 else COLORS['red'] for v in values]
            bars = ax.bar(strategies, values, color=colors, alpha=0.8, edgecolor=COLORS['yellow'], linewidth=2)
            
            for bar in bars:
                bar_color = bar.get_facecolor()
                ax.bar(bar.get_x(), bar.get_height(), bar.get_width(), 
                      color=bar_color, alpha=0.3, edgecolor='none', linewidth=0)
            
            ax.set_title(metric.upper(), fontsize=14, fontweight='bold', family='Rajdhani', pad=15)
            ax.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5)
            
            if len(strategies) > 3:
                ax.set_xticklabels(strategies, rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Comparison table
        st.markdown('<p class="sub-header">Metrics Table</p>', unsafe_allow_html=True)
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

# ==================== TAB 2: MONTHLY PERFORMANCE ====================
with tabs[1]:
    st.markdown('<p class="sub-header">Monthly Performance Analysis</p>', unsafe_allow_html=True)
    
    # Configuration
    if len(dataframes) == 1:
        st.info("Showing monthly returns for single strategy")
        initial_capital = st.number_input("Starting Capital ($)", value=100000, step=10000, min_value=1000)
        weights = None
    else:
        st.markdown("### PORTFOLIO CONFIGURATION")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            initial_capital = st.number_input("Total Capital ($)", value=100000, step=10000, min_value=1000)
        
        with col2:
            st.markdown("**Strategy Weights:**")
            weights = {}
            weight_cols = st.columns(min(len(dataframes), 4))
            
            for idx, name in enumerate(dataframes.keys()):
                with weight_cols[idx % 4]:
                    weights[name] = st.slider(
                        f"{name}", 
                        min_value=0.0, 
                        max_value=2.0, 
                        value=1.0, 
                        step=0.1,
                        key=f"monthly_weight_{name}"
                    )
    
    # Calculate monthly returns
    monthly_df = calculate_monthly_returns(dataframes, weights, initial_capital)
    
    if monthly_df is not None and len(monthly_df) > 0:
        # Year selector with visual buttons
        st.markdown("---")
        available_years = sorted(monthly_df['Year'].unique(), reverse=True)
        
        st.markdown("### MONTHLY PERFORMANCE VISUALIZATION")
        
        # Create year selector buttons
        st.markdown('<p style="color: #A0AEC0; font-family: Rajdhani, sans-serif; margin-bottom: 0.5rem;">Select Year:</p>', unsafe_allow_html=True)
        
        # Use columns for year buttons
        year_cols = st.columns(min(len(available_years), 8))
        selected_year = available_years[0]  # Default to most recent
        
        # Check if there's a stored selection
        if 'selected_year' not in st.session_state:
            st.session_state.selected_year = available_years[0]
        
        # Create button for each year
        for idx, year in enumerate(available_years):
            with year_cols[idx % 8]:
                if st.button(
                    str(int(year)),
                    key=f"year_btn_{year}",
                    use_container_width=True,
                    type="primary" if year == st.session_state.selected_year else "secondary"
                ):
                    st.session_state.selected_year = year
        
        selected_year = st.session_state.selected_year
        
        st.markdown("---")
        
        # Create and display the chart with selected year
        fig = create_monthly_performance_chart(monthly_df, selected_year)
        
        if fig:
            st.pyplot(fig)
        
        # Summary statistics
        st.markdown('<p class="sub-header">Monthly Statistics</p>', unsafe_allow_html=True)
        
        cols = st.columns(5)
        with cols[0]:
            create_metric_card("Best Month", monthly_df['Return_%'].max(), 'percent')
        with cols[1]:
            create_metric_card("Worst Month", monthly_df['Return_%'].min(), 'percent')
        with cols[2]:
            create_metric_card("Avg Month", monthly_df['Return_%'].mean(), 'percent')
        with cols[3]:
            winning_months = len(monthly_df[monthly_df['Return_%'] > 0])
            create_metric_card("Winning Months", f"{winning_months}/{len(monthly_df)}", 'other')
        with cols[4]:
            win_rate = (winning_months / len(monthly_df) * 100) if len(monthly_df) > 0 else 0
            create_metric_card("Monthly Win Rate", win_rate, 'percent')
        
        # Show raw data table
        with st.expander("View Monthly Data Table"):
            display_df = monthly_df.copy()
            display_df['Month_Name'] = pd.to_datetime(display_df['Month'].astype(str), format='%m').dt.strftime('%B')
            display_df = display_df[['Year', 'Month_Name', 'Return_%', 'Net_Return']]
            display_df.columns = ['Year', 'Month', 'Return %', 'Net Return $']
            display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:.2f}%")
            display_df['Net Return $'] = display_df['Net Return $'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No monthly data available. Monthly analysis requires time-series data (Daily, Weekly, Monthly, or Yearly formats).")

# ==================== TAB 3: PERFORMANCE ====================
with tabs[2]:
    st.markdown('<p class="sub-header">Performance Analysis</p>', unsafe_allow_html=True)
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="perf_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    
    if 'Net profit' in df.columns:
        net_profit = df['Net profit'].dropna()
        
        if len(net_profit) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Returns bar chart
            setup_plot_style(fig, ax1)
            
            if format_type == 'day_of_week':
                x_labels = df['Period']
                x_pos = range(len(x_labels))
                colors = [COLORS['green'] if x > 0 else COLORS['red'] for x in net_profit]
                bars = ax1.bar(x_pos, net_profit, color=colors, alpha=0.8, 
                             edgecolor=COLORS['yellow'], linewidth=1.5)
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(x_labels, rotation=0)
                ax1.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
            else:
                colors = [COLORS['green'] if x > 0 else COLORS['red'] for x in net_profit]
                bars = ax1.bar(range(len(net_profit)), net_profit, color=colors, alpha=0.8, 
                             edgecolor=COLORS['yellow'], linewidth=1.5)
                ax1.set_xlabel('Period', fontsize=12, fontweight='bold')
            
            for i, bar in enumerate(bars):
                ax1.bar(bar.get_x(), bar.get_height(), bar.get_width(), 
                       color=bar.get_facecolor(), alpha=0.3, edgecolor='none')
            
            ax1.set_title('PERIOD RETURNS', fontsize=14, fontweight='bold', family='Rajdhani', pad=15)
            ax1.set_ylabel('Net Profit ($)', fontsize=12, fontweight='bold')
            ax1.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5)
            
            # Win/Loss pie
            setup_plot_style(fig, ax2)
            
            wins = net_profit[net_profit > 0]
            losses = net_profit[net_profit < 0]
            
            if len(wins) > 0 or len(losses) > 0:
                categories = []
                values = []
                colors_pie = []
                
                if len(wins) > 0:
                    categories.append(f'Wins\n({len(wins)})')
                    values.append(wins.sum())
                    colors_pie.append(COLORS['green'])
                
                if len(losses) > 0:
                    categories.append(f'Losses\n({len(losses)})')
                    values.append(abs(losses.sum()))
                    colors_pie.append(COLORS['red'])
                
                wedges, texts, autotexts = ax2.pie(values, labels=categories, autopct='%1.1f%%', 
                       colors=colors_pie, startangle=90, textprops={'color': COLORS['white'], 
                       'fontsize': 12, 'fontweight': 'bold'})
                
                for wedge in wedges:
                    wedge.set_edgecolor(COLORS['yellow'])
                    wedge.set_linewidth(2)
                
                ax2.set_title('WIN/LOSS DISTRIBUTION', fontsize=14, fontweight='bold', 
                            family='Rajdhani', pad=15)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Stats
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Total Returns", net_profit.sum(), 'currency')
            with cols[1]:
                create_metric_card("Mean Return", net_profit.mean(), 'currency')
            with cols[2]:
                create_metric_card("Best Period", net_profit.max(), 'currency')
            with cols[3]:
                create_metric_card("Worst Period", net_profit.min(), 'currency')

# ==================== TAB 4: RISK ====================
with tabs[3]:
    st.markdown('<p class="sub-header">Risk Analysis</p>', unsafe_allow_html=True)
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="risk_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    metrics = calculate_summary_metrics(df)
    
    # Risk metrics
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("### DOWNSIDE RISK")
        create_metric_card("Max Drawdown", metrics.get('Max Drawdown', 0), 'currency')
        create_metric_card("Largest Loss", metrics.get('Largest Loser', 0), 'currency')
        create_metric_card("Avg Loss", metrics.get('Avg Loser', 0), 'currency')
    
    with cols[1]:
        st.markdown("### RISK-ADJUSTED")
        create_metric_card("Sharpe Ratio", metrics.get('Sharpe Ratio', 0), 'ratio')
        create_metric_card("Sortino Ratio", metrics.get('Sortino Ratio', 0), 'ratio')
        create_metric_card("Recovery Factor", metrics.get('Recovery Factor', 0), 'ratio')
    
    with cols[2]:
        st.markdown("### EFFICIENCY")
        create_metric_card("Profit Factor", metrics.get('Profit Factor', 0), 'ratio')
        create_metric_card("Win Rate", metrics.get('Win Rate', 0), 'percent')
        create_metric_card("Avg MAE", metrics.get('Avg MAE', 0), 'currency')
    
    # Distribution
    if 'Net profit' in df.columns and format_type != 'day_of_week':
        returns = df['Net profit'].dropna()
        
        if len(returns) > 5:
            st.markdown('<p class="sub-header">Return Distribution</p>', unsafe_allow_html=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Histogram
            setup_plot_style(fig, ax1)
            n, bins, patches = ax1.hist(returns, bins=min(30, len(returns)//2), alpha=0.7, 
                                       color=COLORS['blue'], edgecolor=COLORS['yellow'], linewidth=1.5)
            
            for i, patch in enumerate(patches):
                if bins[i] < 0:
                    patch.set_facecolor(COLORS['red'])
                else:
                    patch.set_facecolor(COLORS['green'])
                patch.set_alpha(0.7)
            
            ax1.axvline(returns.mean(), color=COLORS['yellow'], linestyle='--', linewidth=2.5, 
                       label=f'Mean: ${returns.mean():.2f}', zorder=5)
            ax1.axvline(returns.median(), color=COLORS['yellow_bright'], linestyle='--', linewidth=2.5, 
                       label=f'Median: ${returns.median():.2f}', zorder=5)
            ax1.axvline(0, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5)
            ax1.set_title('RETURN DISTRIBUTION', fontsize=14, fontweight='bold', family='Rajdhani', pad=15)
            ax1.set_xlabel('Net Profit ($)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10, framealpha=0.9)
            
            # Q-Q plot
            setup_plot_style(fig, ax2)
            sorted_returns = np.sort(returns)
            theoretical_quantiles = np.linspace(returns.min(), returns.max(), len(sorted_returns))
            
            ax2.scatter(theoretical_quantiles, sorted_returns, alpha=0.6, s=50, 
                       color=COLORS['yellow'], edgecolors=COLORS['yellow_bright'], linewidths=1)
            ax2.plot([returns.min(), returns.max()], [returns.min(), returns.max()], 
                    color=COLORS['red'], linestyle='--', linewidth=2.5, label='Reference Line')
            ax2.set_title('Q-Q PLOT', fontsize=14, fontweight='bold', family='Rajdhani', pad=15)
            ax2.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Observed Returns', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10, framealpha=0.9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Distribution stats
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Skewness", returns.skew(), 'ratio')
            with cols[1]:
                create_metric_card("Kurtosis", returns.kurtosis(), 'ratio')
            with cols[2]:
                create_metric_card("Std Dev", returns.std(), 'currency')
            with cols[3]:
                downside = returns[returns < 0]
                create_metric_card("Downside Dev", downside.std() if len(downside) > 0 else 0, 'currency')

# ==================== TAB 5: TIME ====================
with tabs[4]:
    st.markdown('<p class="sub-header">Time-Based Analysis</p>', unsafe_allow_html=True)
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="time_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    
    if format_type == 'day_of_week':
        if 'Period' in df.columns and 'Net profit' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 6))
            setup_plot_style(fig, ax)
            
            days = df['Period']
            profits = df['Net profit']
            
            colors = [COLORS['green'] if x > 0 else COLORS['red'] for x in profits]
            bars = ax.bar(days, profits, color=colors, alpha=0.8, edgecolor=COLORS['yellow'], linewidth=2)
            
            for bar in bars:
                ax.bar(bar.get_x(), bar.get_height(), bar.get_width(), 
                      color=bar.get_facecolor(), alpha=0.3, edgecolor='none')
            
            ax.set_title('NET PROFIT BY DAY OF WEEK', fontsize=16, fontweight='bold', 
                        family='Rajdhani', pad=20)
            ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
            ax.set_ylabel('Net Profit ($)', fontsize=12, fontweight='bold')
            ax.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif format_type in ['period', 'trades']:
        if 'Month_Year' in df.columns and 'Net profit' in df.columns:
            monthly = df.groupby('Month_Year')['Net profit'].sum().sort_index()
            
            fig, ax = plt.subplots(figsize=(16, 7))
            setup_plot_style(fig, ax)
            
            colors = [COLORS['green'] if x > 0 else COLORS['red'] for x in monthly.values]
            bars = ax.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.8, 
                         edgecolor=COLORS['yellow'], linewidth=1.5)
            
            for bar in bars:
                ax.bar(bar.get_x(), bar.get_height(), bar.get_width(), 
                      color=bar.get_facecolor(), alpha=0.3, edgecolor='none')
            
            ax.set_title('MONTHLY RETURNS', fontsize=16, fontweight='bold', family='Rajdhani', pad=20)
            ax.set_xlabel('Month', fontsize=12, fontweight='bold')
            ax.set_ylabel('Net Profit ($)', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(monthly)))
            ax.set_xticklabels(monthly.index, rotation=45, ha='right')
            ax.axhline(y=0, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Monthly stats
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Best Month", monthly.max(), 'currency')
            with cols[1]:
                create_metric_card("Worst Month", monthly.min(), 'currency')
            with cols[2]:
                winning_months = len(monthly[monthly > 0])
                create_metric_card("Winning Months", f"{winning_months}/{len(monthly)}", 'other')
            with cols[3]:
                create_metric_card("Avg Monthly", monthly.mean(), 'currency')

# ==================== TAB 6: PORTFOLIO ====================
with tabs[5]:
    st.markdown('<p class="sub-header">Portfolio Analysis</p>', unsafe_allow_html=True)
    
    if len(dataframes) < 2:
        st.info("💡 Upload multiple strategies to enable portfolio analysis")
    else:
        st.markdown("### POSITION SIZING")
        
        # Capital allocation
        total_capital = st.number_input("Total Portfolio Capital ($)", 
                                       value=100000, step=10000, min_value=10000)
        
        st.markdown("**Allocate weights to each strategy:**")
        
        weights = {}
        cols = st.columns(min(len(dataframes), 4))
        
        for idx, name in enumerate(dataframes.keys()):
            with cols[idx % 4]:
                weights[name] = st.slider(
                    f"{name}", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=1.0, 
                    step=0.1,
                    key=f"weight_{name}"
                )
        
        # Calculate portfolio
        portfolio_metrics, combined = calculate_portfolio_metrics(dataframes, weights, total_capital)
        
        if portfolio_metrics and combined is not None:
            st.markdown('<p class="sub-header">Portfolio Performance</p>', unsafe_allow_html=True)
            
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Initial Capital", portfolio_metrics.get('Initial Capital', 0), 'currency')
            with cols[1]:
                create_metric_card("Net Profit", portfolio_metrics.get('Net Profit', 0), 'currency')
            with cols[2]:
                create_metric_card("Total Return", portfolio_metrics.get('Total Return %', 0), 'percent')
            with cols[3]:
                create_metric_card("Sharpe Ratio", portfolio_metrics.get('Sharpe Ratio', 0), 'ratio')
            
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Peak Equity", portfolio_metrics.get('Peak Equity', 0), 'currency')
            with cols[1]:
                create_metric_card("Max Drawdown", portfolio_metrics.get('Max Drawdown', 0), 'currency')
            with cols[2]:
                create_metric_card("Max DD %", portfolio_metrics.get('Max Drawdown %', 0), 'percent')
            with cols[3]:
                create_metric_card("Recovery Factor", portfolio_metrics.get('Recovery Factor', 0), 'ratio')
            
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Profit Factor", portfolio_metrics.get('Profit Factor', 0), 'ratio')
            with cols[1]:
                create_metric_card("Avg Period", portfolio_metrics.get('Avg Period Return', 0), 'currency')
            with cols[2]:
                create_metric_card("Std Dev", portfolio_metrics.get('Std Dev', 0), 'currency')
            with cols[3]:
                create_metric_card("Win Rate", portfolio_metrics.get('Period Win Rate', 0), 'percent')
            
            # Portfolio equity curve
            st.markdown('<p class="sub-header">Portfolio Equity Curve</p>', unsafe_allow_html=True)
            
            monthly_combined = combined.groupby('Month_Year').agg({
                'Portfolio_Cum': 'last'
            }).sort_index()
            
            for name in dataframes.keys():
                cum_col = f'{name}_Cum'
                if cum_col in combined.columns:
                    strategy_monthly = combined.groupby('Month_Year')[cum_col].last().sort_index()
                    monthly_combined[name] = strategy_monthly
            
            # Fetch benchmark if requested
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
                            monthly_combined['Buy & Hold'] = buy_hold_returns.reindex(
                                monthly_combined.index).ffill().fillna(0) + benchmark_capital
                except Exception as e:
                    st.warning(f"Could not fetch benchmark data: {e}")
            
            # Plot with FIXED x-axis labels
            fig, ax = plt.subplots(figsize=(16, 8))
            setup_plot_style(fig, ax)
            
            portfolio_line = ax.plot(monthly_combined.index, monthly_combined['Portfolio_Cum'], 
                   label='Portfolio', linewidth=4, color=COLORS['yellow'], zorder=10)
            ax.plot(monthly_combined.index, monthly_combined['Portfolio_Cum'], 
                   linewidth=8, color=COLORS['yellow'], alpha=0.2, zorder=9)
            
            strategy_colors = [COLORS['blue'], COLORS['green'], COLORS['red'], COLORS['gray']]
            for idx, col in enumerate([c for c in monthly_combined.columns if c not in ['Portfolio_Cum', 'Buy & Hold']]):
                color = strategy_colors[idx % len(strategy_colors)]
                ax.plot(monthly_combined.index, monthly_combined[col], 
                       label=col, linewidth=2, alpha=0.7, color=color, linestyle='--')
            
            if 'Buy & Hold' in monthly_combined.columns:
                ax.plot(monthly_combined.index, monthly_combined['Buy & Hold'], 
                       label='Buy & Hold', linewidth=2.5, linestyle=':', 
                       color=COLORS['red'], alpha=0.8)
            
            ax.set_title('PORTFOLIO EQUITY CURVE - MONTHLY', fontsize=18, fontweight='bold', 
                        family='Rajdhani', pad=20)
            ax.set_xlabel('Month-Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
            ax.axhline(y=total_capital, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5)
            
            # FIXED: Show yearly labels (every 12 months)
            if len(monthly_combined.index) > 12:
                # Get indices for January of each year (or first month if January not present)
                year_indices = []
                seen_years = set()
                for i, month_year in enumerate(monthly_combined.index):
                    year = month_year.split('-')[0]
                    if year not in seen_years:
                        year_indices.append(i)
                        seen_years.add(year)
                
                ax.set_xticks(year_indices)
                ax.set_xticklabels([monthly_combined.index[i] for i in year_indices], rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================== TAB 7: DRAWDOWN ====================
with tabs[6]:
    st.markdown('<p class="sub-header">Drawdown Analysis</p>', unsafe_allow_html=True)
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="dd_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    
    if format_type not in ['period', 'trades']:
        st.info("Drawdown analysis requires time-series data")
    elif 'Cum. net profit' in df.columns:
        cumulative = df['Cum. net profit'].dropna()
        
        if len(cumulative) > 0:
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max
            drawdown_pct = (drawdown / running_max * 100).replace([np.inf, -np.inf], 0).fillna(0)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            
            # Equity with drawdown
            setup_plot_style(fig, ax1)
            x_values = range(len(cumulative))
            ax1.plot(x_values, cumulative.values, label='Equity', linewidth=3, 
                    color=COLORS['yellow'], zorder=3)
            ax1.plot(x_values, cumulative.values, linewidth=6, 
                    color=COLORS['yellow'], alpha=0.2, zorder=2)
            ax1.fill_between(x_values, cumulative.values, running_max.values, 
                            alpha=0.4, color=COLORS['red'], label='Drawdown', zorder=1)
            ax1.plot(x_values, running_max.values, label='Peak Equity', 
                    linewidth=2, linestyle='--', color=COLORS['green'], alpha=0.8)
            ax1.set_title(f'{selected_strategy.upper()} - EQUITY WITH DRAWDOWN', 
                         fontsize=16, fontweight='bold', family='Rajdhani', pad=15)
            ax1.set_ylabel('Profit ($)', fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=11, framealpha=0.9)
            
            # Drawdown chart
            setup_plot_style(fig, ax2)
            ax2.fill_between(x_values, drawdown.values, 0, alpha=0.6, color=COLORS['red'])
            ax2.plot(x_values, drawdown.values, linewidth=2.5, color=COLORS['red_glow'])
            ax2.set_title('DRAWDOWN OVER TIME', fontsize=16, fontweight='bold', 
                         family='Rajdhani', pad=15)
            ax2.set_xlabel('Period', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Drawdown ($)', fontsize=12, fontweight='bold')
            ax2.axhline(y=drawdown.min(), color=COLORS['yellow'], linestyle='--', 
                       linewidth=2, alpha=0.9, label=f'Max DD: ${drawdown.min():,.2f}')
            ax2.legend(fontsize=11, framealpha=0.9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Stats
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Max Drawdown", drawdown.min(), 'currency')
            with cols[1]:
                max_dd_pct = drawdown_pct.min()
                create_metric_card("Max DD %", max_dd_pct if not np.isnan(max_dd_pct) else 0, 'percent')
            with cols[2]:
                create_metric_card("Current DD", drawdown.iloc[-1], 'currency')
            with cols[3]:
                in_drawdown = drawdown < 0
                current_dd_duration = 0
                for is_dd in reversed(in_drawdown.values):
                    if is_dd:
                        current_dd_duration += 1
                    else:
                        break
                create_metric_card("Current DD Duration", f"{current_dd_duration}", 'other')

# ==================== TAB 8: PERIODS ====================
with tabs[7]:
    st.markdown('<p class="sub-header">Period-by-Period Analysis</p>', unsafe_allow_html=True)
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="period_select")
    df = dataframes[selected_strategy]
    
    if 'Net profit' in df.columns:
        returns = df['Net profit'].dropna()
        
        if len(returns) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Histogram
            setup_plot_style(fig, ax1)
            n, bins, patches = ax1.hist(returns, bins=min(30, max(10, len(returns)//3)), 
                                       alpha=0.7, color=COLORS['blue'], 
                                       edgecolor=COLORS['yellow'], linewidth=1.5)
            
            for i, patch in enumerate(patches):
                if bins[i] < 0:
                    patch.set_facecolor(COLORS['red'])
                else:
                    patch.set_facecolor(COLORS['green'])
                patch.set_alpha(0.7)
            
            ax1.axvline(returns.mean(), color=COLORS['yellow'], linestyle='--', 
                       linewidth=2.5, label=f'Mean: ${returns.mean():.2f}')
            ax1.axvline(0, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5)
            ax1.set_title('RETURN DISTRIBUTION', fontsize=14, fontweight='bold', 
                         family='Rajdhani', pad=15)
            ax1.set_xlabel('Net Profit ($)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10, framealpha=0.9)
            
            # Box plot
            setup_plot_style(fig, ax2)
            bp = ax2.boxplot(returns, vert=True, patch_artist=True,
                       boxprops=dict(facecolor=COLORS['blue'], alpha=0.7, 
                                   edgecolor=COLORS['yellow'], linewidth=2),
                       medianprops=dict(color=COLORS['yellow'], linewidth=3),
                       whiskerprops=dict(color=COLORS['yellow'], linewidth=1.5),
                       capprops=dict(color=COLORS['yellow'], linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor=COLORS['red'], 
                                     markersize=8, markeredgecolor=COLORS['yellow']))
            ax2.set_title('BOX PLOT', fontsize=14, fontweight='bold', family='Rajdhani', pad=15)
            ax2.set_ylabel('Net Profit ($)', fontsize=12, fontweight='bold')
            ax2.axhline(0, color=COLORS['gray'], linestyle='-', linewidth=1, alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Mean Return", returns.mean(), 'currency')
            with cols[1]:
                create_metric_card("Median Return", returns.median(), 'currency')
            with cols[2]:
                create_metric_card("Std Dev", returns.std(), 'currency')
            with cols[3]:
                profitable = len(returns[returns > 0])
                create_metric_card("Profitable Periods", f"{profitable}/{len(returns)}", 'other')
    
    # Data table
    st.markdown('<p class="sub-header">Period Data</p>', unsafe_allow_html=True)
    display_cols = ['Period', 'Cum. net profit', 'Net profit', '% Win', 'Avg. trade', 
                   'Max. drawdown', 'Gross profit', 'Gross loss']
    display_cols = [col for col in display_cols if col in df.columns]
    
    st.dataframe(df[display_cols].tail(20), use_container_width=True)
    
    if st.button("📥 Download Full Period Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Click to Download CSV",
            data=csv,
            file_name=f"{selected_strategy}_periods.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #A0AEC0; font-family: Rajdhani, sans-serif;'>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
            <strong style='color: #FFD700;'>NinjaTrader Analytics Terminal</strong>
        </p>
        <p style='font-size: 0.9rem;'>
            Professional Trading Analytics • Multi-Strategy Portfolio Analysis • Real-Time Performance Metrics
        </p>
    </div>
""", unsafe_allow_html=True)
