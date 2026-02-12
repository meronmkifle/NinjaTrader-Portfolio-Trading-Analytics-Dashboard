import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from matplotlib import font_manager
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Enterprise Trading Analytics",
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DESIGN SYSTEM ====================

COLORS = {
    'primary_dark': '#0F172A',      # Deep navy
    'primary_mid': '#1E293B',       # Medium navy
    'primary_light': '#334155',     # Light slate
    'accent_cyan': '#06B6D4',       # Cyan accent
    'accent_lime': '#84CC16',       # Lime accent
    'success': '#10B981',            # Green
    'warning': '#F59E0B',            # Amber
    'danger': '#EF4444',             # Red
    'white': '#FFFFFF',
    'gray_light': '#E2E8F0',
    'gray_dark': '#475569',
    'text_primary': '#F1F5F9',
    'text_secondary': '#CBD5E1',
    'background_dark': '#0B1117',
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {COLORS['background_dark']} 0%, {COLORS['primary_dark']} 100%);
        background-attachment: fixed;
    }}
    
    .main {{
        background: transparent;
    }}
    
    /* Headers */
    h1, h2, h3, h4 {{
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    .header-main {{
        font-size: 2.8rem;
        background: linear-gradient(135deg, {COLORS['accent_cyan']} 0%, {COLORS['accent_lime']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }}
    
    .header-sub {{
        font-size: 1.3rem;
        color: {COLORS['text_primary']};
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid {COLORS['accent_cyan']};
        display: inline-block;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['primary_mid']} 0%, {COLORS['primary_light']} 100%);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 
                    0 0 1px rgba(6, 182, 212, 0.2);
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, {COLORS['accent_cyan']}, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(6, 182, 212, 0.5);
        box-shadow: 0 12px 40px rgba(6, 182, 212, 0.15), 
                    0 0 20px rgba(6, 182, 212, 0.1);
    }}
    
    .metric-card:hover::before {{
        opacity: 1;
    }}
    
    .metric-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
        opacity: 0.85;
    }}
    
    .metric-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: {COLORS['accent_cyan']};
        line-height: 1.2;
        letter-spacing: -0.5px;
    }}
    
    .metric-value.positive {{
        color: {COLORS['success']};
    }}
    
    .metric-value.negative {{
        color: {COLORS['danger']};
    }}
    
    .metric-value.neutral {{
        color: {COLORS['accent_lime']};
    }}
    
    .metric-delta {{
        font-size: 0.75rem;
        color: {COLORS['text_secondary']};
        margin-top: 0.25rem;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['primary_dark']} 0%, {COLORS['primary_mid']} 100%);
        border-right: 1px solid rgba(6, 182, 212, 0.1);
    }}
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {COLORS['accent_cyan']};
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {COLORS['text_secondary']};
        font-size: 0.9rem;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0px;
        background: transparent;
        padding: 0;
        border-bottom: 2px solid rgba(6, 182, 212, 0.1);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: {COLORS['text_secondary']};
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 1rem 1.5rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 2px solid transparent;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['accent_cyan']};
        border-bottom-color: rgba(6, 182, 212, 0.3);
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {COLORS['accent_cyan']} !important;
        border-bottom-color: {COLORS['accent_cyan']} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        background: linear-gradient(135deg, {COLORS['accent_cyan']} 0%, {COLORS['accent_lime']} 100%);
        color: {COLORS['primary_dark']};
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.4);
    }}
    
    /* Input elements */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div {{
        background-color: {COLORS['primary_light']};
        border-color: rgba(6, 182, 212, 0.2);
        color: {COLORS['text_primary']};
    }}
    
    /* Dataframe */
    .dataframe {{
        font-family: 'IBM Plex Mono', monospace !important;
        background: {COLORS['primary_mid']} !important;
        border: 1px solid rgba(6, 182, 212, 0.1) !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
    }}
    
    .dataframe th {{
        background: {COLORS['primary_light']} !important;
        color: {COLORS['accent_cyan']} !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        border: none !important;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['primary_dark']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {COLORS['accent_cyan']} 0%, {COLORS['accent_lime']} 100%);
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['accent_cyan']};
    }}
    
    /* Messages */
    .stSuccess, .stInfo, .stWarning {{
        background: linear-gradient(135deg, {COLORS['primary_mid']} 0%, {COLORS['primary_light']} 100%);
        border-left: 4px solid {COLORS['accent_cyan']};
        color: {COLORS['text_primary']};
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }}
    
    .stError {{
        background: linear-gradient(135deg, {COLORS['primary_mid']} 0%, {COLORS['primary_light']} 100%);
        border-left: 4px solid {COLORS['danger']};
        color: {COLORS['text_primary']};
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        color: {COLORS['accent_cyan']};
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# Set matplotlib style
plt.style.use('dark_background')
sns.set_palette([COLORS['accent_cyan'], COLORS['success'], COLORS['accent_lime'], COLORS['danger']])

def setup_plot_style(fig, ax):
    """Apply professional matplotlib styling"""
    fig.patch.set_facecolor(COLORS['primary_mid'])
    ax.set_facecolor(COLORS['primary_dark'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['primary_light'])
        spine.set_linewidth(0.8)
    
    ax.tick_params(colors=COLORS['text_secondary'], which='both', labelsize=9)
    ax.xaxis.label.set_color(COLORS['text_primary'])
    ax.yaxis.label.set_color(COLORS['text_primary'])
    ax.title.set_color(COLORS['text_primary'])
    ax.title.set_fontweight('bold')
    
    ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5, color=COLORS['accent_cyan'])

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
    """Clean currency columns"""
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
    """Parse NinjaTrader files with format auto-detection"""
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
        
        if format_type in ['period', 'trades'] and 'Period' in df.columns:
            df = df.sort_values('Period').reset_index(drop=True)
        elif format_type == 'day_of_week' and 'Day_Order' in df.columns:
            df = df.sort_values('Day_Order').reset_index(drop=True)
        
        df.attrs['format_type'] = format_type
        return df
    
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

# ==================== ADVANCED METRICS ====================

def calculate_advanced_metrics(df):
    """Calculate comprehensive metrics including advanced statistics"""
    metrics = {}
    format_type = df.attrs.get('format_type', 'period')
    
    if len(df) == 0:
        return metrics
    
    # Basic metrics
    if format_type == 'day_of_week':
        metrics['Net Profit'] = df['Net profit'].sum() if 'Net profit' in df.columns else 0
        metrics['Gross Profit'] = df['Gross profit'].sum() if 'Gross profit' in df.columns else 0
        metrics['Gross Loss'] = abs(df['Gross loss'].sum()) if 'Gross loss' in df.columns else 0
    else:
        last_row = df.iloc[-1]
        metrics['Net Profit'] = last_row.get('Cum. net profit', 0)
        if pd.isna(metrics['Net Profit']):
            metrics['Net Profit'] = df['Net profit'].sum() if 'Net profit' in df.columns else 0
        
        metrics['Gross Profit'] = df['Gross profit'].sum() if 'Gross profit' in df.columns else 0
        metrics['Gross Loss'] = abs(df['Gross loss'].sum()) if 'Gross loss' in df.columns else 0
    
    # Profit Factor
    metrics['Profit Factor'] = metrics['Gross Profit'] / metrics['Gross Loss'] if metrics['Gross Loss'] != 0 else 0
    
    # Trade metrics
    metrics['Avg Trade'] = df['Avg. trade'].mean() if 'Avg. trade' in df.columns else 0
    metrics['Avg Winner'] = df['Avg. winner'].mean() if 'Avg. winner' in df.columns else 0
    metrics['Avg Loser'] = df['Avg. loser'].mean() if 'Avg. loser' in df.columns else 0
    metrics['Win Rate'] = df['% Win'].mean() if '% Win' in df.columns else 0
    metrics['Largest Winner'] = df['Lrg. winner'].max() if 'Lrg. winner' in df.columns else 0
    metrics['Largest Loser'] = df['Lrg. loser'].min() if 'Lrg. loser' in df.columns else 0
    
    # Drawdown metrics
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
    
    # Risk-adjusted returns
    if 'Net profit' in df.columns and format_type != 'day_of_week':
        returns = df['Net profit'].dropna()
        if len(returns) > 1 and returns.std() > 0:
            metrics['Sharpe Ratio'] = (returns.mean() / returns.std() * np.sqrt(252))
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
            metrics['Sortino Ratio'] = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
            
            metrics['Calmar Ratio'] = (returns.mean() * 252) / abs(metrics['Max Drawdown']) if metrics['Max Drawdown'] != 0 else 0
            metrics['Return Volatility'] = returns.std()
        else:
            metrics['Sharpe Ratio'] = 0
            metrics['Sortino Ratio'] = 0
            metrics['Calmar Ratio'] = 0
            metrics['Return Volatility'] = 0
    
    # Recovery metrics
    if metrics.get('Max Drawdown', 0) != 0:
        metrics['Recovery Factor'] = abs(metrics['Net Profit'] / metrics['Max Drawdown'])
    else:
        metrics['Recovery Factor'] = 0
    
    # Execution metrics
    metrics['Avg MAE'] = df['Avg. MAE'].mean() if 'Avg. MAE' in df.columns else 0
    metrics['Avg MFE'] = df['Avg. MFE'].mean() if 'Avg. MFE' in df.columns else 0
    metrics['Total Periods'] = len(df)
    
    if 'Net profit' in df.columns:
        profitable_periods = len(df[df['Net profit'] > 0])
        metrics['Profitable Periods'] = profitable_periods
        metrics['Period Win Rate'] = (profitable_periods / len(df) * 100) if len(df) > 0 else 0
    
    # Additional risk metrics
    if 'Net profit' in df.columns:
        returns = df['Net profit'].dropna()
        if len(returns) > 0:
            metrics['Skewness'] = returns.skew()
            metrics['Kurtosis'] = returns.kurtosis()
            metrics['Max Consecutive Wins'] = calculate_max_consecutive_wins(df)
            metrics['Max Consecutive Losses'] = calculate_max_consecutive_losses(df)
            metrics['RRR'] = calculate_risk_reward_ratio(df)  # Risk/Reward Ratio
    
    return metrics

def calculate_max_consecutive_wins(df):
    """Calculate maximum consecutive winning periods"""
    if 'Net profit' not in df.columns:
        return 0
    
    wins = (df['Net profit'] > 0).astype(int)
    max_consecutive = 0
    current_consecutive = 0
    
    for win in wins:
        if win:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def calculate_max_consecutive_losses(df):
    """Calculate maximum consecutive losing periods"""
    if 'Net profit' not in df.columns:
        return 0
    
    losses = (df['Net profit'] < 0).astype(int)
    max_consecutive = 0
    current_consecutive = 0
    
    for loss in losses:
        if loss:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def calculate_risk_reward_ratio(df):
    """Calculate average risk/reward ratio"""
    if 'Avg. winner' not in df.columns or 'Avg. loser' not in df.columns:
        return 0
    
    avg_winner = df['Avg. winner'].mean()
    avg_loser = abs(df['Avg. loser'].mean())
    
    if avg_loser == 0:
        return 0
    
    return avg_winner / avg_loser

def calculate_monthly_returns(dataframes, weights=None, initial_capital=100000):
    """Calculate monthly returns for portfolio"""
    valid_dfs = {name: df for name, df in dataframes.items() 
                 if df.attrs.get('format_type', 'period') in ['period', 'trades']}
    
    if not valid_dfs:
        return None
    
    if weights is None:
        weights = {name: 1.0 for name in valid_dfs.keys()}
    
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    monthly_data = {}
    
    for name, df in valid_dfs.items():
        if 'Period' in df.columns and 'Net profit' in df.columns:
            df_copy = df.copy()
            df_copy['Year'] = pd.to_datetime(df_copy['Period']).dt.year
            df_copy['Month'] = pd.to_datetime(df_copy['Period']).dt.month
            
            monthly_grouped = df_copy.groupby(['Year', 'Month'])['Net profit'].sum()
            
            for (year, month), net_profit in monthly_grouped.items():
                key = (year, month)
                if key not in monthly_data:
                    monthly_data[key] = 0
                
                monthly_data[key] += net_profit * weights[name]
    
    monthly_df = pd.DataFrame([
        {
            'Year': year,
            'Month': month,
            'Net_Return': ret
        }
        for (year, month), ret in monthly_data.items()
    ])
    
    monthly_df['Return_%'] = (monthly_df['Net_Return'] / initial_capital) * 100
    monthly_df = monthly_df.sort_values(['Year', 'Month'])
    
    return monthly_df

def format_metric(value, metric_type='currency'):
    """Format metrics for display"""
    if pd.isna(value):
        return "N/A"
    if metric_type == 'currency':
        return f"${value:,.2f}"
    elif metric_type == 'percent':
        return f"{value:.2f}%"
    elif metric_type == 'ratio':
        return f"{value:.2f}x"
    else:
        return f"{value:.2f}"

def create_metric_card(label, value, value_type='currency', delta=None):
    """Create styled metric card"""
    if value_type == 'currency':
        display_value = format_metric(value, 'currency')
        value_class = 'positive' if value > 0 else 'negative' if value < 0 else 'neutral'
    elif value_type == 'percent':
        display_value = format_metric(value, 'percent')
        value_class = 'positive' if value > 0 else 'negative' if value < 0 else 'neutral'
    elif value_type == 'ratio':
        display_value = format_metric(value, 'ratio')
        value_class = 'positive' if value > 1 else 'negative' if value < 1 else 'neutral'
    else:
        display_value = str(value)
        value_class = 'neutral'
    
    delta_html = ""
    if delta is not None:
        delta_class = 'positive' if delta > 0 else 'negative' if delta < 0 else 'neutral'
        delta_html = f'<div class="metric-delta" style="color: {COLORS["text_secondary"]};">Δ {delta:+.2f}%</div>'
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {value_class}">{display_value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="header-main">📊 Trading Analytics Suite</h1>', unsafe_allow_html=True)

st.markdown('<p style="text-align: center; color: #CBD5E1; font-size: 0.95rem;">Professional-Grade Performance Analytics for Algorithmic Trading Strategies</p>', 
           unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<h2 class="header-sub">Data Input</h2>', unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader(
    "Upload NinjaTrader Exports", 
    type=['csv', 'xlsx'],
    accept_multiple_files=True,
    help="CSV or Excel files from NinjaTrader strategy analyzer"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<h2 class="header-sub">Benchmarking</h2>', unsafe_allow_html=True)
show_benchmark = st.sidebar.checkbox("Compare to Benchmark", value=False)
benchmark_ticker = None
benchmark_capital = 100000

if show_benchmark:
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="ES=F", help="e.g., ES=F, NQ=F, SPY")
    benchmark_capital = st.sidebar.number_input("Benchmark Capital ($)", value=100000, step=10000)

# Process files
dataframes = {}
if uploaded_files:
    st.sidebar.markdown("---")
    st.sidebar.markdown('<h2 class="header-sub">Loaded Strategies</h2>', unsafe_allow_html=True)
    for file in uploaded_files:
        df = parse_ninjatrader_file(file)
        if df is not None:
            strategy_name = file.name.replace('.csv', '').replace('.xlsx', '')
            dataframes[strategy_name] = df
            format_type = df.attrs.get('format_type', 'unknown')
            st.sidebar.success(f"✓ {strategy_name} ({format_type})")

if not dataframes:
    st.markdown("""
        <div style='text-align: center; padding: 6rem 2rem;'>
            <h2 style='color: #06B6D4; font-family: Inter, sans-serif; font-size: 2rem; margin-bottom: 1.5rem;'>
                Welcome to Trading Analytics Suite
            </h2>
            <p style='color: #CBD5E1; font-family: Inter, sans-serif; font-size: 1.1rem; line-height: 1.8; max-width: 600px; margin: 0 auto;'>
                Upload your NinjaTrader strategy exports to begin comprehensive performance analysis.<br><br>
                <strong style='color: #06B6D4;'>Supported Formats:</strong><br>
                • Daily, Weekly, Monthly, Yearly Performance Summaries<br>
                • Day of Week Analysis<br>
                • Individual Trade Executions<br><br>
                <strong style='color: #84CC16;'>Features:</strong><br>
                • Advanced risk metrics and drawdown analysis<br>
                • Multi-strategy portfolio optimization<br>
                • Monthly performance heatmaps<br>
                • Benchmark comparisons<br>
                • Professional visualizations
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Create tabs
tabs = st.tabs(["DASHBOARD", "MONTHLY", "ANALYSIS", "RISK", "PORTFOLIO", "STATISTICS"])

# ==================== TAB 1: DASHBOARD ====================
with tabs[0]:
    st.markdown('<h2 class="header-sub">Performance Dashboard</h2>', unsafe_allow_html=True)
    
    if len(dataframes) == 1:
        strategy_name = list(dataframes.keys())[0]
        df = dataframes[strategy_name]
        format_type = df.attrs.get('format_type', 'period')
        metrics = calculate_advanced_metrics(df)
        
        st.markdown(f"**{strategy_name}** • {format_type.replace('_', ' ').title()}")
        
        # Key metrics grid
        st.markdown("### Core Metrics")
        cols = st.columns(4)
        
        with cols[0]:
            create_metric_card("Net Profit", metrics.get('Net Profit', 0), 'currency')
        with cols[1]:
            create_metric_card("Profit Factor", metrics.get('Profit Factor', 0), 'ratio')
        with cols[2]:
            create_metric_card("Win Rate", metrics.get('Win Rate', 0), 'percent')
        with cols[3]:
            create_metric_card("Sharpe Ratio", metrics.get('Sharpe Ratio', 0), 'ratio')
        
        st.markdown("### Risk Metrics")
        cols = st.columns(4)
        
        with cols[0]:
            create_metric_card("Max Drawdown", metrics.get('Max Drawdown', 0), 'currency')
        with cols[1]:
            create_metric_card("Sortino Ratio", metrics.get('Sortino Ratio', 0), 'ratio')
        with cols[2]:
            create_metric_card("Recovery Factor", metrics.get('Recovery Factor', 0), 'ratio')
        with cols[3]:
            create_metric_card("Calmar Ratio", metrics.get('Calmar Ratio', 0), 'ratio')
        
        st.markdown("### Trade Metrics")
        cols = st.columns(4)
        
        with cols[0]:
            create_metric_card("Avg Trade", metrics.get('Avg Trade', 0), 'currency')
        with cols[1]:
            create_metric_card("Avg Winner", metrics.get('Avg Winner', 0), 'currency')
        with cols[2]:
            create_metric_card("Avg Loser", metrics.get('Avg Loser', 0), 'currency')
        with cols[3]:
            create_metric_card("Risk/Reward", metrics.get('RRR', 0), 'ratio')
        
        # Equity curve
        if format_type in ['period', 'trades'] and 'Cum. net profit' in df.columns:
            st.markdown("### Equity Curve")
            
            fig, ax = plt.subplots(figsize=(16, 6))
            setup_plot_style(fig, ax)
            
            cumulative = df['Cum. net profit'].dropna()
            x_values = range(len(cumulative))
            
            ax.plot(x_values, cumulative.values, linewidth=3, color=COLORS['accent_cyan'], 
                   label='Equity', zorder=3)
            ax.fill_between(x_values, 0, cumulative.values, alpha=0.15, color=COLORS['accent_cyan'])
            
            if format_type == 'trades':
                ax.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
            else:
                ax.set_xlabel('Period', fontsize=11, fontweight='bold')
                if 'Period' in df.columns:
                    n_ticks = min(10, len(df))
                    tick_indices = np.linspace(0, len(df)-1, n_ticks, dtype=int)
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([df.iloc[i]['Period'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['Period'], 'strftime') else str(df.iloc[i]['Period']) for i in tick_indices], rotation=45, ha='right')
            
            ax.set_title(f'{strategy_name.upper()} - Equity Evolution', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('Cumulative Profit ($)', fontsize=11, fontweight='bold')
            ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=0.8, alpha=0.3)
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        # Multiple strategies comparison
        st.markdown("### Strategy Comparison")
        
        all_metrics = {}
        for name, df in dataframes.items():
            all_metrics[name] = calculate_advanced_metrics(df)
        
        # Comparison heatmap
        comparison_metrics = ['Net Profit', 'Profit Factor', 'Win Rate', 'Sharpe Ratio', 
                            'Max Drawdown', 'Sortino Ratio', 'Recovery Factor', 'Calmar Ratio']
        
        comparison_data = []
        for metric in comparison_metrics:
            row = {'Metric': metric}
            for name in all_metrics.keys():
                value = all_metrics[name].get(metric, 0)
                row[name] = value
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        setup_plot_style(fig, ax)
        
        # Normalize for heatmap
        metric_names = comparison_df['Metric'].values
        data_for_heatmap = comparison_df.drop('Metric', axis=1).T.values
        
        im = ax.imshow(data_for_heatmap, cmap='RdYlGn', aspect='auto', alpha=0.8)
        
        ax.set_xticks(range(len(metric_names)))
        ax.set_yticks(range(len(all_metrics)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(all_metrics.keys())
        
        # Add values to cells
        for i in range(len(all_metrics)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{data_for_heatmap[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        ax.set_title('Strategy Performance Heatmap', fontsize=14, fontweight='bold', pad=15)
        fig.colorbar(im, ax=ax, label='Value')
        
        plt.tight_layout()
        st.pyplot(fig)

# ==================== TAB 2: MONTHLY PERFORMANCE ====================
with tabs[1]:
    st.markdown('<h2 class="header-sub">Monthly Performance Analysis</h2>', unsafe_allow_html=True)
    
    if len(dataframes) == 1:
        st.info("Single strategy monthly analysis")
        initial_capital = st.number_input("Starting Capital ($)", value=100000, step=10000, min_value=1000)
    else:
        col1, col2 = st.columns([1, 2])
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
    
    monthly_df = calculate_monthly_returns(dataframes, None if len(dataframes) == 1 else weights, initial_capital)
    
    if monthly_df is not None and len(monthly_df) > 0:
        # Create enhanced monthly heatmap
        st.markdown("### Monthly Returns Heatmap")
        
        pivot_data = monthly_df.pivot(index='Year', columns='Month', values='Return_%')
        pivot_data = pivot_data.reindex(columns=range(1, 13))
        pivot_data = pivot_data.sort_index(ascending=False)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        setup_plot_style(fig, ax)
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5, alpha=0.9)
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(range(12))
        ax.set_yticks(range(len(pivot_data)))
        ax.set_xticklabels(months)
        ax.set_yticklabels([int(year) for year in pivot_data.index])
        
        # Add text annotations
        for i in range(len(pivot_data)):
            for j in range(12):
                value = pivot_data.iloc[i, j]
                if not pd.isna(value):
                    color = 'white' if abs(value) > 3 else 'black'
                    text = ax.text(j, i, f'{value:.1f}%', ha="center", va="center", 
                                 color=color, fontsize=9, fontweight='bold')
        
        ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Month', fontsize=11, fontweight='bold')
        ax.set_ylabel('Year', fontsize=11, fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax, label='Return %')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary statistics
        st.markdown("### Monthly Statistics")
        cols = st.columns(5)
        
        with cols[0]:
            create_metric_card("Best Month", monthly_df['Return_%'].max(), 'percent')
        with cols[1]:
            create_metric_card("Worst Month", monthly_df['Return_%'].min(), 'percent')
        with cols[2]:
            create_metric_card("Avg Month", monthly_df['Return_%'].mean(), 'percent')
        with cols[3]:
            winning_months = len(monthly_df[monthly_df['Return_%'] > 0])
            create_metric_card(f"Winning Months", f"{winning_months}/{len(monthly_df)}", 'other')
        with cols[4]:
            win_rate = (winning_months / len(monthly_df) * 100) if len(monthly_df) > 0 else 0
            create_metric_card("Monthly Win Rate", win_rate, 'percent')

# ==================== TAB 3: ANALYSIS ====================
with tabs[2]:
    st.markdown('<h2 class="header-sub">Detailed Performance Analysis</h2>', unsafe_allow_html=True)
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="analysis_select")
    df = dataframes[selected_strategy]
    format_type = df.attrs.get('format_type', 'period')
    
    if 'Net profit' in df.columns:
        net_profit = df['Net profit'].dropna()
        
        if len(net_profit) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # Returns distribution
            ax = axes[0, 0]
            setup_plot_style(fig, ax)
            colors = [COLORS['success'] if x > 0 else COLORS['danger'] for x in net_profit]
            ax.bar(range(len(net_profit)), net_profit, color=colors, alpha=0.7, edgecolor=COLORS['accent_cyan'], linewidth=0.5)
            ax.set_title('Period Returns', fontsize=12, fontweight='bold')
            ax.set_ylabel('Net Profit ($)', fontsize=10, fontweight='bold')
            ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=0.8, alpha=0.3)
            
            # Histogram
            ax = axes[0, 1]
            setup_plot_style(fig, ax)
            ax.hist(net_profit, bins=min(30, len(net_profit)//2), color=COLORS['accent_cyan'], alpha=0.7, edgecolor=COLORS['accent_lime'], linewidth=1)
            ax.axvline(net_profit.mean(), color=COLORS['accent_lime'], linestyle='--', linewidth=2.5, label=f'Mean: ${net_profit.mean():.2f}')
            ax.axvline(0, color=COLORS['text_secondary'], linestyle='-', linewidth=0.8, alpha=0.3)
            ax.set_title('Return Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Net Profit ($)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.legend(fontsize=9, framealpha=0.9)
            
            # Cumulative returns
            ax = axes[1, 0]
            setup_plot_style(fig, ax)
            cumulative_returns = net_profit.cumsum()
            ax.plot(range(len(cumulative_returns)), cumulative_returns.values, linewidth=2.5, color=COLORS['accent_cyan'])
            ax.fill_between(range(len(cumulative_returns)), 0, cumulative_returns.values, alpha=0.15, color=COLORS['accent_cyan'])
            ax.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cumulative Profit ($)', fontsize=10, fontweight='bold')
            ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=0.8, alpha=0.3)
            
            # Drawdown
            ax = axes[1, 1]
            setup_plot_style(fig, ax)
            if 'Cum. net profit' in df.columns:
                cumulative = df['Cum. net profit'].dropna()
                running_max = cumulative.cummax()
                drawdown = cumulative - running_max
                ax.fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.6, color=COLORS['danger'])
                ax.plot(range(len(drawdown)), drawdown.values, linewidth=2, color=COLORS['danger'])
            ax.set_title('Drawdown Profile', fontsize=12, fontweight='bold')
            ax.set_ylabel('Drawdown ($)', fontsize=10, fontweight='bold')
            ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', linewidth=0.8, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# ==================== TAB 4: RISK ====================
with tabs[3]:
    st.markdown('<h2 class="header-sub">Risk Analysis</h2>', unsafe_allow_html=True)
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="risk_select")
    df = dataframes[selected_strategy]
    metrics = calculate_advanced_metrics(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Downside Risk")
        create_metric_card("Max Drawdown", metrics.get('Max Drawdown', 0), 'currency')
        create_metric_card("Largest Loss", metrics.get('Largest Loser', 0), 'currency')
        create_metric_card("Return Volatility", metrics.get('Return Volatility', 0), 'currency')
    
    with col2:
        st.markdown("#### Risk-Adjusted Returns")
        create_metric_card("Sharpe Ratio", metrics.get('Sharpe Ratio', 0), 'ratio')
        create_metric_card("Sortino Ratio", metrics.get('Sortino Ratio', 0), 'ratio')
        create_metric_card("Calmar Ratio", metrics.get('Calmar Ratio', 0), 'ratio')
    
    with col3:
        st.markdown("#### Consistency")
        create_metric_card("Max Cons. Wins", f"{metrics.get('Max Consecutive Wins', 0)}", 'other')
        create_metric_card("Max Cons. Losses", f"{metrics.get('Max Consecutive Losses', 0)}", 'other')
        create_metric_card("Period Win Rate", metrics.get('Period Win Rate', 0), 'percent')

# ==================== TAB 5: PORTFOLIO ====================
with tabs[4]:
    st.markdown('<h2 class="header-sub">Portfolio Analytics</h2>', unsafe_allow_html=True)
    
    if len(dataframes) < 2:
        st.info("Upload multiple strategies to enable portfolio analysis")
    else:
        st.markdown("### Position Sizing")
        
        total_capital = st.number_input("Total Portfolio Capital ($)", 
                                       value=100000, step=10000, min_value=10000)
        
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
                    key=f"port_weight_{name}"
                )
        
        st.markdown("### Portfolio Performance")
        
        # Calculate combined metrics
        all_metrics = {name: calculate_advanced_metrics(dataframes[name]) for name in dataframes.keys()}
        
        cols = st.columns(4)
        with cols[0]:
            avg_sharpe = np.mean([m.get('Sharpe Ratio', 0) for m in all_metrics.values()])
            create_metric_card("Avg Sharpe", avg_sharpe, 'ratio')
        with cols[1]:
            avg_sortino = np.mean([m.get('Sortino Ratio', 0) for m in all_metrics.values()])
            create_metric_card("Avg Sortino", avg_sortino, 'ratio')
        with cols[2]:
            total_net_profit = sum([m.get('Net Profit', 0) for m in all_metrics.values()])
            create_metric_card("Total Net Profit", total_net_profit, 'currency')
        with cols[3]:
            avg_profit_factor = np.mean([m.get('Profit Factor', 0) for m in all_metrics.values()])
            create_metric_card("Avg Profit Factor", avg_profit_factor, 'ratio')

# ==================== TAB 6: STATISTICS ====================
with tabs[5]:
    st.markdown('<h2 class="header-sub">Statistical Analysis</h2>', unsafe_allow_html=True)
    
    selected_strategy = st.selectbox("Select Strategy", list(dataframes.keys()), key="stats_select")
    df = dataframes[selected_strategy]
    metrics = calculate_advanced_metrics(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Distribution Properties")
        create_metric_card("Skewness", metrics.get('Skewness', 0), 'ratio')
        create_metric_card("Kurtosis", metrics.get('Kurtosis', 0), 'ratio')
    
    with col2:
        st.markdown("#### Trade Characteristics")
        create_metric_card("Total Periods", f"{metrics.get('Total Periods', 0)}", 'other')
        create_metric_card("Avg MAE", metrics.get('Avg MAE', 0), 'currency')
    
    with col3:
        st.markdown("#### Performance Summary")
        create_metric_card("Largest Winner", metrics.get('Largest Winner', 0), 'currency')
        create_metric_card("Largest Loser", metrics.get('Largest Loser', 0), 'currency')
    
    # Distribution visualization
    if 'Net profit' in df.columns:
        returns = df['Net profit'].dropna()
        
        if len(returns) > 5:
            st.markdown("### Return Distribution Analysis")
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # KDE + Histogram
            ax = axes[0]
            setup_plot_style(fig, ax)
            ax.hist(returns, bins=min(30, len(returns)//2), color=COLORS['accent_cyan'], alpha=0.6, edgecolor=COLORS['accent_lime'], linewidth=1, density=True)
            
            # KDE
            try:
                kde = gaussian_kde(returns)
                x_range = np.linspace(returns.min(), returns.max(), 200)
                ax.plot(x_range, kde(x_range), linewidth=2.5, color=COLORS['accent_lime'], label='KDE')
            except:
                pass
            
            ax.axvline(returns.mean(), color=COLORS['accent_lime'], linestyle='--', linewidth=2, label=f'Mean')
            ax.axvline(returns.median(), color=COLORS['warning'], linestyle='--', linewidth=2, label=f'Median')
            ax.set_title('Distribution with KDE', fontsize=12, fontweight='bold')
            ax.set_xlabel('Net Profit ($)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Density', fontsize=10, fontweight='bold')
            ax.legend(fontsize=9, framealpha=0.9)
            
            # Q-Q Plot
            ax = axes[1]
            setup_plot_style(fig, ax)
            sorted_returns = np.sort(returns)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
            
            ax.scatter(theoretical_quantiles, sorted_returns, alpha=0.6, s=40, color=COLORS['accent_cyan'], edgecolors=COLORS['accent_lime'], linewidths=0.8)
            
            # Add reference line
            min_val, max_val = sorted_returns.min(), sorted_returns.max()
            ax.plot([min_val, max_val], [min_val, max_val], color=COLORS['danger'], linestyle='--', linewidth=2.5, label='Reference')
            
            ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='bold')
            ax.set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold')
            ax.legend(fontsize=9, framealpha=0.9)
            
            plt.tight_layout()
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; padding: 2rem; color: {COLORS["text_secondary"]}; font-family: Inter, sans-serif;'>
        <p style='font-size: 1rem; margin-bottom: 0.5rem;'>
            <strong style='color: {COLORS["accent_cyan"]}; font-size: 1.1rem;'>Trading Analytics Suite</strong>
        </p>
        <p style='font-size: 0.85rem; opacity: 0.7;'>
            Professional-grade analytics for algorithmic trading strategies • Multi-strategy portfolio analysis • Real-time performance metrics
        </p>
    </div>
""", unsafe_allow_html=True)
