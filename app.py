import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import xgboost as xgb
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
import requests
import shap
import joblib
import warnings
import asyncio
import websockets
warnings.filterwarnings('ignore')

class AdvancedTradeRiskModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models_trained = False

    def fetch_market_data(self, symbol, period='1mo', interval='1h'):
        api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={st.secrets["TWELVEDATA_API_KEY"]}'
        response = requests.get(api_url)
        data = response.json()
        if 'values' not in data:
            return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.astype(float)
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        df['MACD'] = MACD(close=df['close']).macd()
        return df

    def train_models(self, market_data):
        features = ['close', 'volume', 'VWAP', 'ATR', 'MACD']
        X = market_data[features].fillna(method='ffill')
        y = market_data['close'].pct_change().shift(-1).fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.xgb_model.fit(X_scaled, y)
        self.rf_model.fit(X_scaled, y)
        self.models_trained = True

    def predict_price_movement(self, market_data):
        if not self.models_trained:
            self.train_models(market_data)
        features = ['close', 'volume', 'VWAP', 'ATR', 'MACD']
        X = market_data[features].fillna(method='ffill')
        X_scaled = self.scaler.transform(X)
        xgb_pred = self.xgb_model.predict(X_scaled[-1:])
        rf_pred = self.rf_model.predict(X_scaled[-1:])
        return (xgb_pred[0] + rf_pred[0]) / 2



st.set_page_config(page_title='Advanced Trade Risk Analytics', layout='wide')
st.title("🚀 Advanced Trade Risk Analytics Platform")
model = AdvancedTradeRiskModel()

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("Stock Symbol", "AAPL")
with col2:
    period = st.selectbox("Analysis Period", ["1d", "5d", "1mo", "3mo"], index=2)
with col3:
    trade_size = st.number_input("Trade Size (shares)", min_value=1, value=100)

if st.button("Analyze Risk"):
    with st.spinner("Fetching market data..."):
        market_data = model.fetch_market_data(symbol, period=period)
        if market_data is None:
            st.error("No data found for the given symbol.")
        else:
            model.train_models(market_data)
            risk_prediction = model.predict_price_movement(market_data)
            st.metric("Predicted Price Movement:", f"{risk_prediction:.2%}")
            st.plotly_chart(px.line(market_data, x=market_data.index, y=['close', 'VWAP'], title="Price & VWAP"), use_container_width=True)
            
