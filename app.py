import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model import AdvancedTradeRiskModel
import os
import emoji

# Configure page
st.set_page_config(
    page_title="Advanced Trade Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/trade-risk-analytics',
        'Report a bug': "https://github.com/yourusername/trade-risk-analytics/issues",
        'About': "# Trade Risk Analytics\n This application provides advanced risk analysis for trading decisions."
    }
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-high {
        color: red;
        font-weight: bold;
    }
    .risk-medium {
        color: orange;
        font-weight: bold;
    }
    .risk-low {
        color: green;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def create_gauge_chart(risk_score, title="Risk Score"):
    """Create an enhanced gauge chart with multiple zones"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 0.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.2], 'color': 'green'},
                {'range': [0.2, 0.4], 'color': 'lightgreen'},
                {'range': [0.4, 0.6], 'color': 'yellow'},
                {'range': [0.6, 0.8], 'color': 'orange'},
                {'range': [0.8, 1], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def create_risk_breakdown(risk_metrics):
    """Create enhanced risk component visualization"""
    metrics = {k: v for k, v in risk_metrics.items() if k != 'total_risk'}
    
    fig = px.bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        title="Risk Component Analysis",
        color=list(metrics.values()),
        color_continuous_scale="RdYlGn_r"
    )
    
    fig.update_layout(
        xaxis_title="Risk Components",
        yaxis_title="Risk Level",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def plot_price_history(market_data):
    """Create interactive price history chart"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=market_data.index,
        open=market_data['Open'],
        high=market_data['High'],
        low=market_data['Low'],
        close=market_data['Close'],
        name="Price"
    ))
    
    # Add technical indicators
    if 'SMA' in market_data.columns:
        fig.add_trace(go.Scatter(
            x=market_data.index,
            y=market_data['SMA'],
            name="20-day SMA",
            line=dict(color='blue', width=1)
        ))
    
    if 'BB_upper' in market_data.columns and 'BB_lower' in market_data.columns:
        fig.add_trace(go.Scatter(
            x=market_data.index,
            y=market_data['BB_upper'],
            name="BB Upper",
            line=dict(color='gray', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=market_data.index,
            y=market_data['BB_lower'],
            name="BB Lower",
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ))
    
    fig.update_layout(
        title="Price History with Technical Indicators",
        yaxis_title="Price",
        height=500,
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )
    return fig

def show_stock_examples(exchange):
    """Display example stock symbols with enhanced formatting"""
    if exchange == "India (NSE)":
        examples = {
            "TCS.NS": "Tata Consultancy Services",
            "RELIANCE.NS": "Reliance Industries",
            "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank",
            "WIPRO.NS": "Wipro",
            "TATAMOTORS.NS": "Tata Motors"
        }
    else:
        examples = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms Inc.",
            "TSLA": "Tesla Inc."
        }
    
    st.markdown("### Example Symbols")
    for symbol, company in examples.items():
        st.markdown(f"**{symbol}** - {company}")

def validate_symbol(symbol, exchange):
    """Validate trading symbol"""
    symbol = symbol.strip().upper()
    
    if exchange == "India (NSE)":
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
    else:
        symbol = symbol.replace('.NS', '')
    
    return symbol

def display_metrics(market_data, latest_data):
    """Display key market metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price",
            f"${latest_data['Close']:.2f}",
            f"{latest_data['Price_Change']:.2%}"
        )
        st.metric(
            "Volume",
            f"{latest_data['Volume']:,.0f}",
            f"{(latest_data['Volume']/latest_data['Volume_MA']-1):.1%}" if 'Volume_MA' in latest_data else None
        )
    
    with col2:
        st.metric(
            "Volatility",
            f"{latest_data['Volatility']:.2%}"
        )
        if 'RSI' in latest_data:
            st.metric(
                "RSI",
                f"{latest_data['RSI']:.1f}",
                delta_color="inverse"
            )
    
    with col3:
        st.metric(
            "Spread",
            f"${latest_data['Spread']:.2f}"
        )
        if 'BB_upper' in latest_data and 'BB_lower' in latest_data:
            bb_width = (latest_data['BB_upper'] - latest_data['BB_lower']) / latest_data['Close']
            st.metric(
                "Bollinger Band Width",
                f"{bb_width:.2%}"
            )

def main():
    try:
        st.title("🚀 Advanced Trade Risk Analytics Platform")
        st.markdown("### AI-Powered Trading Risk Assessment")
        
        # Initialize model
        model = AdvancedTradeRiskModel()
        
        # Sidebar configurations
        with st.sidebar:
            st.header("Analysis Settings")
            analysis_period = st.selectbox(
                "Analysis Period",
                ["1d", "5d", "1mo", "3mo"],
                index=1,
                help="Select the period for analysis"
            )
            
            show_technical = st.checkbox("Show Technical Indicators", value=True)
            show_predictions = st.checkbox("Show Price Predictions", value=True)
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("""
            This platform provides advanced risk analysis for trading decisions using:
            - Machine Learning Models
            - Technical Indicators
            - Market Metrics
            """)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            exchange = st.selectbox(
                "Select Exchange",
                ["US", "India (NSE)"],
                index=0
            )
            
            default_symbol = "AAPL" if exchange == "US" else "TCS"
            symbol = st.text_input(
                "Stock Symbol", 
                default_symbol,
                help="Enter the stock symbol (e.g., AAPL for Apple Inc.)"
            )
            
            # Validate and format symbol
            symbol = validate_symbol(symbol, exchange)
            
            trade_size = st.number_input(
                "Trade Size (shares)", 
                min_value=1, 
                value=100,
                help="Enter the number of shares to trade"
            )
            
            st.caption(f"Using symbol: {symbol}")
        
        with col2:
            show_stock_examples(exchange)
        
        if st.button("Analyze Risk", type="primary"):
            try:
                with st.spinner(f"Fetching market data for {symbol}..."):
                    market_data = model.fetch_market_data(
                        symbol=symbol,
                        period=analysis_period
                    )
                    
                    if market_data is None or market_data.empty:
                        st.error(f"No data available for {symbol}. Please verify the symbol.")
                        return
                
                with st.spinner("Calculating risk metrics..."):
                    risk_metrics = model.calculate_risk_metrics(market_data, trade_size)
                    recommendations = model.get_trade_recommendations(risk_metrics, trade_size)
                    optimal_time = model.get_optimal_execution_time(market_data)
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Market Data", "Recommendations"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(create_gauge_chart(risk_metrics['total_risk']), use_container_width=True)
                        with col2:
                            st.plotly_chart(create_risk_breakdown(risk_metrics), use_container_width=True)
                    
                    with tab2:
                        if show_technical:
                            st.plotly_chart(plot_price_history(market_data), use_container_width=True)
                        
                        latest_data = market_data.iloc[-1]
                        display_metrics(market_data, latest_data)
                    
                    with tab3:
                        st.subheader("Trading Recommendations")
                        
                        # Risk level with colored badge
                        risk_color = {
                            "High": "risk-high",
                            "Medium-High": "risk-medium",
                            "Medium": "risk-medium",
                            "Low": "risk-low"
                        }
                        
                        st.markdown(f"""
                        - **Risk Level:** <span class="{risk_color[recommendations['risk_level']]}">{recommendations['risk_level']}</span>
                        - **Recommended Action:** {recommendations['action']}
                        - **Suggested Position Size:** {recommendations['suggested_size']:,} shares
                        - **Optimal Execution Time:** {optimal_time}
                        - **Confidence Score:** {recommendations['confidence']:.1%}
                        """, unsafe_allow_html=True)
                        
                        if recommendations['risk_level'] in ['High', 'Medium-High']:
                            st.warning("⚠️ High risk detected. Consider reducing position size or waiting for better conditions.")
                        
                        # Market status
                        st.subheader("Market Information")
                        market_status = "Open" if model.is_market_open(exchange) else "Closed"
                        market_hours = "9:15 AM - 3:30 PM IST" if exchange == "India (NSE)" else "9:30 AM - 4:00 PM EST"
                        
                        st.info(f"""
                        - Market Status: {market_status}
                        - Trading Hours: {market_hours}
                        - Analysis Period: {analysis_period}
                        """)
                
            except Exception as e:
                error_msg = str(e)
                if "not found" in error_msg.lower():
                    st.error(f"Symbol {symbol} not found. Please verify the symbol.")
                    if exchange == "US":
                        st.info("For US stocks, don't add .NS suffix")
                    else:
                        st.info("For Indian stocks, make sure to use NSE symbols")
                elif "market" in error_msg.lower():
                    st.warning(error_msg)
                    st.info("Try analyzing a different time period or wait for market hours")
                else:
                    st.error(f"Error analyzing {symbol}: {error_msg}")
                    st.info("Please try again with different parameters")
                
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
