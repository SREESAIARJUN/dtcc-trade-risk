import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradeRiskModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.xgb_model = None
        self.lstm_model = None
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.initialize_models()

    def initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize sentiment analysis model
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')

            # Initialize LSTM model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 6)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            self.lstm_model.compile(optimizer='adam', loss='mse')
        except Exception as e:
            print(f"Model initialization error: {str(e)}")

    def is_market_open(self, exchange="US"):
        now = datetime.now()
        if exchange == "India":
            india_time = now + timedelta(hours=5, minutes=30)
            market_open = india_time.replace(hour=9, minute=15)
            market_close = india_time.replace(hour=15, minute=30)
            return market_open <= india_time <= market_close
        else:
            est_time = now - timedelta(hours=5)
            market_open = est_time.replace(hour=9, minute=30)
            market_close = est_time.replace(hour=16, minute=0)
            return market_open <= est_time <= market_close

    def convert_to_usd(self, price, from_currency="INR"):
        if from_currency == "INR":
            usd_inr_rate = 83.0
            return price / usd_inr_rate
        return price

    def fetch_market_data(self, symbol, period='1d', interval='1m'):
        try:
            is_indian = symbol.endswith('.NS')
            exchange = "India" if is_indian else "US"
            
            if not self.is_market_open(exchange):
                market_hours = "9:15 AM - 3:30 PM IST" if is_indian else "9:30 AM - 4:00 PM EST"
                raise ValueError(f"{exchange} markets are closed (Trading hours: {market_hours})")

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            if is_indian:
                for col in ['Open', 'High', 'Low', 'Close']:
                    data[col] = data[col].apply(lambda x: self.convert_to_usd(x, "INR"))
                
            return self.process_market_data(data)
        except Exception as e:
            raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

    def process_market_data(self, data):
        if data.empty:
            raise ValueError("No market data available to process")
            
        data['Volatility'] = data['Close'].rolling(window=20).std()
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Price_Change'] = data['Close'].pct_change()
        data['Spread'] = data['High'] - data['Low']
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'] = self.calculate_macd(data['Close'])
        return data

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def prepare_lstm_data(self, data, lookback=60):
        features = ['Close', 'Volume', 'Volatility', 'Volume_MA', 'Price_Change', 'Spread']
        scaled_data = self.price_scaler.fit_transform(data[features])
        
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:(i + lookback)])
            y.append(scaled_data[i + lookback, 0])
        return np.array(X), np.array(y)

    def train_models(self, market_data):
        if len(market_data) < 100:
            return

        features = ['Volatility', 'Volume', 'Spread', 'Price_Change', 'RSI', 'MACD']
        X = market_data[features].fillna(0)
        y = (market_data['Close'].pct_change() > 0).astype(int)
        
        self.xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=3
        )
        self.xgb_model.fit(X, y)

        X_lstm, y_lstm = self.prepare_lstm_data(market_data)
        if len(X_lstm) > 0:
            self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)

        self.anomaly_detector.fit(X)

    def detect_anomalies(self, market_data):
        features = ['Volatility', 'Volume', 'Spread', 'Price_Change', 'RSI', 'MACD']
        X = market_data[features].fillna(0)
        anomalies = self.anomaly_detector.predict(X)
        return np.mean(anomalies == -1)

    def predict_price_movement(self, market_data):
        if self.lstm_model is None:
            return 0

        X_lstm, _ = self.prepare_lstm_data(market_data)
        if len(X_lstm) > 0:
            prediction = self.lstm_model.predict(X_lstm[-1:])
            return self.price_scaler.inverse_transform(prediction)[0][0]
        return 0

    def calculate_risk_metrics(self, market_data, trade_size):
        if market_data.empty:
            return {
                'volatility_risk': 1.0,
                'liquidity_risk': 1.0,
                'spread_risk': 1.0,
                'volume_risk': 1.0,
                'technical_risk': 1.0,
                'ml_risk': 1.0,
                'total_risk': 1.0
            }

        latest_data = market_data.iloc[-1]
        avg_volume = market_data['Volume'].mean()

        # Calculate technical indicators risk
        rsi_risk = abs(latest_data['RSI'] - 50) / 50
        macd_risk = abs(latest_data['MACD']) / market_data['MACD'].std()

        risk_metrics = {
            'volatility_risk': min(1, latest_data['Volatility'] / latest_data['Close']),
            'liquidity_risk': min(1, trade_size / avg_volume if avg_volume != 0 else 1),
            'spread_risk': min(1, latest_data['Spread'] / latest_data['Close']),
            'volume_risk': min(1, 1 - (latest_data['Volume'] / latest_data['Volume_MA'])),
            'technical_risk': min(1, (rsi_risk + macd_risk) / 2),
            'ml_risk': min(1, self.detect_anomalies(market_data))
        }

        weights = {
            'volatility_risk': 0.2,
            'liquidity_risk': 0.2,
            'spread_risk': 0.15,
            'volume_risk': 0.15,
            'technical_risk': 0.15,
            'ml_risk': 0.15
        }

        total_risk = sum(risk_metrics[k] * weights[k] for k in weights.keys())
        risk_metrics['total_risk'] = min(1, total_risk)

        return risk_metrics

    def get_trade_recommendations(self, risk_metrics, trade_size):
        total_risk = risk_metrics['total_risk']
        
        if total_risk > 0.8:
            risk_level = "High"
            action = "Avoid trading or reduce position size significantly"
            suggested_size = trade_size * 0.3
        elif total_risk > 0.6:
            risk_level = "Medium-High"
            action = "Consider splitting trade into smaller orders"
            suggested_size = trade_size * 0.5
        elif total_risk > 0.4:
            risk_level = "Medium"
            action = "Proceed with caution"
            suggested_size = trade_size * 0.75
        else:
            risk_level = "Low"
            action = "Proceed with trade"
            suggested_size = trade_size

        return {
            'risk_level': risk_level,
            'action': action,
            'suggested_size': int(suggested_size),
            'original_size': trade_size,
            'warnings': self.generate_warnings(risk_metrics)
        }

    def generate_warnings(self, risk_metrics):
        warnings = []
        if risk_metrics['volatility_risk'] > 0.7:
            warnings.append("High market volatility detected")
        if risk_metrics['liquidity_risk'] > 0.7:
            warnings.append("Low market liquidity")
        if risk_metrics['ml_risk'] > 0.7:
            warnings.append("ML models indicate high risk")
        return warnings

    def analyze_market_sentiment(self, symbol):
        try:
            # Placeholder for sentiment analysis
            return np.array([0.3, 0.4, 0.3])  # [negative, neutral, positive]
        except:
            return None
