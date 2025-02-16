import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from model import TradeRiskModel
import time

st.set_page_config(page_title="Trade Settlement Risk Predictor", layout="wide")

def create_gauge_chart(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.3], 'color': "green"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'bar': {'color': "darkblue"}
        }
    ))
    return fig

def create_risk_breakdown(risk_metrics):
    # Remove total_risk for breakdown chart
    metrics = {k: v for k, v in risk_metrics.items() if k != 'total_risk'}
    
    fig = px.bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        title="Risk Breakdown"
    )
    fig.update_layout(
        xaxis_title="Risk Components",
        yaxis_title="Risk Level",
        yaxis_range=[0, 1]
    )
    return fig

def main():
    st.title("🚀 Trade Settlement Risk Predictor")
    
    # Initialize model
    model = TradeRiskModel()
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", "AAPL")
        trade_size = st.number_input("Trade Size (shares)", 
                                   min_value=1, 
                                   value=100)
    
    # Analysis button
    if st.button("Analyze Risk"):
        with st.spinner("Fetching market data and analyzing risk..."):
            # Fetch and analyze data
            market_data = model.fetch_market_data(symbol)
            risk_metrics = model.calculate_risk_metrics(market_data, trade_size)
            recommendations = model.get_trade_recommendations(risk_metrics, trade_size)
            optimal_time = model.get_optimal_execution_time(market_data)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(risk_metrics['total_risk']))
                
                # Risk level indicator
                if risk_metrics['total_risk'] > 0.7:
                    st.error(f"⚠️ High Risk Level: {risk_metrics['total_risk']:.2%}")
                elif risk_metrics['total_risk'] > 0.3:
                    st.warning(f"⚠️ Medium Risk Level: {risk_metrics['total_risk']:.2%}")
                else:
                    st.success(f"✅ Low Risk Level: {risk_metrics['total_risk']:.2%}")
            
            with col2:
                st.plotly_chart(create_risk_breakdown(risk_metrics))
            
            # Recommendations section
            st.subheader("📊 Trade Recommendations")
            st.write(f"Risk Level: {recommendations['risk_level']}")
            st.write(f"Recommended Action: {recommendations['action']}")
            st.write(f"Suggested Trade Size: {recommendations['suggested_size']:,} shares")
            st.write(f"Optimal Execution Time: {optimal_time}")
            
            # Market conditions
            st.subheader("📈 Current Market Conditions")
            latest_data = market_data.iloc[-1]
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Current Price", f"${latest_data['Close']:.2f}")
                st.metric("Volume", f"{latest_data['Volume']:,}")
            
            with metrics_col2:
                st.metric("Volatility", f"{latest_data['Volatility']:.2%}")
                st.metric("Spread", f"${latest_data['Spread']:.2f}")

if __name__ == "__main__":
    main()
