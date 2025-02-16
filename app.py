import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from model import AdvancedTradeRiskModel
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Advanced Trade Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_gauge_chart(risk_score):
    """Enhanced gauge chart with more detailed risk zones"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={'text': "Risk Score"},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.2], 'color': "darkgreen"},
                {'range': [0.2, 0.4], 'color': "lightgreen"},
                {'range': [0.4, 0.6], 'color': "yellow"},
                {'range': [0.6, 0.8], 'color': "orange"},
                {'range': [0.8, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            },
            'bar': {'color': "darkblue"}
        }
    ))
    return fig

def create_risk_breakdown(risk_metrics):
    """Enhanced risk breakdown with ML components"""
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
        showlegend=False
    )
    return fig

def plot_price_prediction(market_data, predictions):
    """Plot historical prices and predictions"""
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=market_data.index,
        y=market_data['Close'],
        name="Historical Price",
        line=dict(color='blue')
    ))
    
    # Predictions
    pred_dates = pd.date_range(
        start=market_data.index[-1],
        periods=len(predictions)+1,
        freq='1D'
    )[1:]
    
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=predictions,
        name="Predicted Price",
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Price History and Predictions",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )
    return fig

def plot_anomaly_detection(market_data, anomalies):
    """Plot price chart with anomaly highlights"""
    fig = go.Figure()
    
    # Normal points
    normal_data = market_data[anomalies == 1]
    fig.add_trace(go.Scatter(
        x=normal_data.index,
        y=normal_data['Close'],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=6)
    ))
    
    # Anomaly points
    anomaly_data = market_data[anomalies == -1]
    fig.add_trace(go.Scatter(
        x=anomaly_data.index,
        y=anomaly_data['Close'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title="Anomaly Detection Results",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    return fig

def display_sentiment_analysis(sentiment_scores):
    """Display sentiment analysis results"""
    if sentiment_scores is None:
        st.warning("Sentiment analysis data unavailable")
        return
        
    sentiment_df = pd.DataFrame({
        'Sentiment': ['Negative', 'Neutral', 'Positive'],
        'Score': sentiment_scores
    })
    
    fig = px.pie(
        sentiment_df,
        values='Score',
        names='Sentiment',
        title='Market Sentiment Analysis',
        color_discrete_sequence=['red', 'gray', 'green']
    )
    return fig

def main():
    st.title("🚀 Advanced Trade Risk Analytics Platform")
    st.markdown("### AI-Powered Trading Risk Assessment")
    
    # Initialize advanced model
    model = AdvancedTradeRiskModel()
    
    # Sidebar configurations
    with st.sidebar:
        st.header("Analysis Settings")
        analysis_period = st.selectbox(
            "Analysis Period",
            ["1d", "5d", "1mo", "3mo"],
            index=1
        )
        
        prediction_days = st.slider(
            "Prediction Horizon (Days)",
            1, 30, 5
        )
        
        show_anomalies = st.checkbox("Show Anomaly Detection", value=True)
        show_predictions = st.checkbox("Show Price Predictions", value=True)
        show_sentiment = st.checkbox("Show Sentiment Analysis", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        exchange = st.selectbox(
            "Select Exchange",
            ["US", "India (NSE)"],
            index=0
        )
        
        symbol = st.text_input(
            "Stock Symbol", 
            "AAPL" if exchange == "US" else "TCS"
        ).upper()
        
        if exchange == "India (NSE)":
            symbol = f"{symbol}.NS"
            
        trade_size = st.number_input(
            "Trade Size (shares)", 
            min_value=1, 
            value=100
        )
    
    with col2:
        show_stock_examples(exchange)
    
    if st.button("Analyze Risk", type="primary"):
        try:
            with st.spinner("Performing advanced analysis..."):
                # Fetch and process data
                market_data = model.fetch_market_data(symbol, period=analysis_period)
                
                if market_data.empty:
                    st.error(f"No data available for {symbol}")
                    return
                
                # Train models with historical data
                model.train_models(market_data)
                
                # Calculate risk metrics and predictions
                risk_metrics = model.calculate_risk_metrics(market_data, trade_size)
                predictions = model.predict_price_movement(market_data)
                anomalies = model.detect_anomalies(market_data)
                sentiment_scores = model.analyze_market_sentiment(symbol)
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Technical Analysis", "Market Sentiment"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_gauge_chart(risk_metrics['total_risk']))
                    with col2:
                        st.plotly_chart(create_risk_breakdown(risk_metrics))
                
                with tab2:
                    if show_predictions:
                        st.plotly_chart(plot_price_prediction(market_data, predictions))
                    if show_anomalies:
                        st.plotly_chart(plot_anomaly_detection(market_data, anomalies))
                
                with tab3:
                    if show_sentiment:
                        st.plotly_chart(display_sentiment_analysis(sentiment_scores))
                
                # Trading recommendations
                st.subheader("📊 AI-Powered Trading Recommendations")
                recommendations = model.get_trade_recommendations(risk_metrics, trade_size)
                
                for key, value in recommendations.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")
            st.info("Please verify the symbol and try again.")

if __name__ == "__main__":
    main()
