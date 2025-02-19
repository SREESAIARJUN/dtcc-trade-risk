import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import xgboost as xgb
from ta.volatility import AverageTrueRange
from ta.trend import MACD
import requests
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# Style Enhancements
# -------------------------------------------------------------------------
st.set_page_config(page_title='Advanced Trade Risk Analytics', layout='wide')
st.markdown("""
    <style>
    .main { 
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Model Class
# -------------------------------------------------------------------------
class AdvancedTradeRiskModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=100
        )
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.models_trained = False

    def fetch_market_data(self, symbol, period='1mo', interval='1h'):
        """
        Fetch market data from TwelveData API.
        """
        api_key = st.secrets["TWELVEDATA_API_KEY"]
        api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}'
        response = requests.get(api_url)
        data = response.json()
        if 'values' not in data:
            return None

        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        # Convert columns to floats
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = df[col].astype(float)
        
        # Additional indicators
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        df['MACD'] = MACD(close=df['close']).macd()

        # Only keep relevant data within the selected period
        # Simplistic approach: for '1d','5d','1mo','3mo', interpret as day range
        now = df.index.max()
        if period == '1d':
            cutoff = now - timedelta(days=1)
        elif period == '5d':
            cutoff = now - timedelta(days=5)
        elif period == '3mo':
            cutoff = now - timedelta(days=90)
        else:
            # fallback for '1mo'
            cutoff = now - timedelta(days=30)
        df = df[df.index >= cutoff]

        return df.sort_index()

    def train_models(self, market_data):
        """
        Train both XGBoost and RandomForest models using current market data.
        """
        features = ['close', 'volume', 'VWAP', 'ATR', 'MACD']
        X = market_data[features].fillna(method='ffill')
        y = market_data['close'].pct_change().shift(-1).fillna(0)

        X_scaled = self.scaler.fit_transform(X)
        self.xgb_model.fit(X_scaled, y)
        self.rf_model.fit(X_scaled, y)
        self.models_trained = True

    def predict_price_movement(self, market_data):
        """
        Use the trained models to predict price movement (percentage change) 
        for the most recent bar.
        """
        if not self.models_trained:
            self.train_models(market_data)
        features = ['close', 'volume', 'VWAP', 'ATR', 'MACD']
        X = market_data[features].fillna(method='ffill')
        X_scaled = self.scaler.transform(X)
        xgb_pred = self.xgb_model.predict(X_scaled[-1:])
        rf_pred = self.rf_model.predict(X_scaled[-1:])
        return (xgb_pred[0] + rf_pred[0]) / 2

    def calculate_risk_metrics(self, market_data, trade_size):
        """
        Calculate various risk metrics that feed into the total risk score.
        """
        latest_data = market_data.iloc[-1]
        avg_volume = market_data['volume'].mean()

        # Sample risk metrics, normalized between 0 and 1
        volatility_risk = min(1, latest_data['ATR'] / latest_data['close'])
        liquidity_risk = min(1, trade_size / avg_volume if avg_volume != 0 else 1)
        spread_risk = min(1, (latest_data['high'] - latest_data['low']) / latest_data['close'])
        price_movement_risk = abs(self.predict_price_movement(market_data))

        risk_metrics = {
            'Volatility Risk': volatility_risk,
            'Liquidity Risk': liquidity_risk,
            'Spread Risk': spread_risk,
            'Price Movement Risk': price_movement_risk
        }
        total_risk = sum(risk_metrics.values()) / len(risk_metrics)
        risk_metrics['Total Risk'] = total_risk
        return risk_metrics

    def generate_trade_recommendation(self, risk_score):
        """
        Provide a text recommendation based on the overall risk score.
        """
        if risk_score > 0.80:
            return ("High risk detected. "
                    "You may consider reducing your position size or using tighter stops. "
                    "Review market volatility and liquidity before entering.")
        elif risk_score > 0.60:
            return ("Medium-high risk. "
                    "Splitting your trade, hedging, or waiting for calmer market conditions might be prudent.")
        elif risk_score > 0.40:
            return ("Moderate risk. "
                    "Proceed with caution, ensure proper risk management (like stop-loss orders).")
        else:
            return ("Low risk detected. "
                    "You can proceed with your planned trade, but always monitor the market for unexpected changes.")

# -------------------------------------------------------------------------
# Streamlit App Layout
# -------------------------------------------------------------------------
st.title("🚀 Advanced Trade Risk Analytics Platform")

model = AdvancedTradeRiskModel()

# ---------------------------
# Input Fields
# ---------------------------
with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Stock Symbol", "AAPL")
    period = st.selectbox("Analysis Period", ["1d", "5d", "1mo", "3mo"], index=2)
    trade_size = st.number_input("Trade Size (shares)", min_value=1, value=100)

analyze_button = st.button("Analyze Risk")

# ---------------------------
# Main Execution
# ---------------------------
if analyze_button:
    with st.spinner("Fetching market data and analyzing..."):
        market_data = model.fetch_market_data(symbol, period=period)
        if market_data is None or market_data.empty:
            st.error("No data found for the given symbol or period.")
        else:
            model.train_models(market_data)
            risk_metrics = model.calculate_risk_metrics(market_data, trade_size)
            recommendation = model.generate_trade_recommendation(risk_metrics['Total Risk'])
            predicted_movement = model.predict_price_movement(market_data)

            # Provide feedback balloons
            st.balloons()

            # ---------------------------
            # Tabs for better organization
            # ---------------------------
            tab1, tab2, tab3 = st.tabs(["Data Preview", "Risk Analysis", "Recommendations"])

            # ---------------------------
            # Tab 1: Data Preview
            # ---------------------------
            with tab1:
                st.subheader(f"{symbol} Market Data Preview")
                st.write(market_data.tail(10))

                # Plot the close price
                fig_price = px.line(
                    market_data, 
                    x=market_data.index, 
                    y='close', 
                    title=f"{symbol} Closing Price Over Time"
                )
                st.plotly_chart(fig_price, use_container_width=True)

                # Optionally, plot volume or other indicator
                fig_volume = px.bar(
                    market_data, 
                    x=market_data.index, 
                    y='volume', 
                    title=f"{symbol} Volume Over Time"
                )
                st.plotly_chart(fig_volume, use_container_width=True)

            # ---------------------------
            # Tab 2: Risk Analysis
            # ---------------------------
            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted 1-Bar Price Movement", f"{predicted_movement:.2%}")
                    st.metric("Total Risk Score", f"{risk_metrics['Total Risk']:.2f}")

                with col2:
                    # Bar chart for risk breakdown
                    risk_keys = list(risk_metrics.keys())
                    risk_values = list(risk_metrics.values())
                    fig_risk = px.bar(
                        x=risk_keys[:-1],  # exclude Total Risk from the bars
                        y=risk_values[:-1],
                        title="Risk Breakdown",
                        color=risk_values[:-1],
                        color_continuous_scale="RdYlGn_r"
                    )
                    fig_risk.update_layout(xaxis_title="", yaxis_title="Risk Level (0-1)")
                    st.plotly_chart(fig_risk, use_container_width=True)

            # ---------------------------
            # Tab 3: Recommendations
            # ---------------------------
            with tab3:
                st.subheader("Trade Recommendation")
                st.info(recommendation)

                # Further details
                st.markdown("""  
                <b>Practical Tips:</b>  
                • If volatility risk is high, consider placing tighter stops or reducing position size.  
                • If liquidity risk is high, be cautious about large orders that might slip.  
                • Spread risk can increase costs; check limit orders vs. market orders.  
                • Always combine model insights with fundamental analysis and market context.  
                """, unsafe_allow_html=True)

else:
    st.write("Configure your trade parameters on the left and click 'Analyze Risk' to get started.")

# -------------------------------------------------------------------------
# Disclaimer
# -------------------------------------------------------------------------
st.caption("Disclaimer: This tool is for educational and informational purposes only and does not constitute financial advice. "+
           "Always do your own research before making any investment or trading decisions.")
