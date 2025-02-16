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

    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        features = ['Close', 'Volume', 'Volatility', 'Volume_MA', 'Price_Change', 'Spread']
        scaled_data = self.price_scaler.fit_transform(data[features])
        
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:(i + lookback)])
            y.append(scaled_data[i + lookback, 0])
        return np.array(X), np.array(y)

    def train_models(self, market_data):
        """Train ML models with market data"""
        if len(market_data) < 100:
            return

        # Train XGBoost for risk prediction
        features = ['Volatility', 'Volume', 'Spread', 'Price_Change']
        X = market_data[features].fillna(0)
        y = (market_data['Close'].pct_change() > 0).astype(int)
        
        self.xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=3
        )
        self.xgb_model.fit(X, y)

        # Train LSTM for price prediction
        X_lstm, y_lstm = self.prepare_lstm_data(market_data)
        if len(X_lstm) > 0:
            self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)

        # Train anomaly detector
        self.anomaly_detector.fit(X)

    def detect_anomalies(self, market_data):
        """Detect market anomalies"""
        features = ['Volatility', 'Volume', 'Spread', 'Price_Change']
        X = market_data[features].fillna(0)
        anomalies = self.anomaly_detector.predict(X)
        return np.mean(anomalies == -1)  # Return anomaly ratio

    def predict_price_movement(self, market_data):
        """Predict future price movement"""
        if self.lstm_model is None:
            return 0

        X_lstm, _ = self.prepare_lstm_data(market_data)
        if len(X_lstm) > 0:
            prediction = self.lstm_model.predict(X_lstm[-1:])
            return self.price_scaler.inverse_transform(prediction)[0][0]
        return 0

    def calculate_risk_metrics(self, market_data, trade_size):
        """Enhanced risk metrics calculation"""
        basic_metrics = super().calculate_risk_metrics(market_data, trade_size)
        
        # Add ML-based risk factors
        anomaly_risk = self.detect_anomalies(market_data)
        
        if self.xgb_model is not None:
            features = ['Volatility', 'Volume', 'Spread', 'Price_Change']
            X = market_data[features].fillna(0).iloc[-1:]
            ml_risk = 1 - self.xgb_model.predict_proba(X)[0][1]
        else:
            ml_risk = 0.5

        # Combine traditional and ML-based metrics
        enhanced_metrics = {
            **basic_metrics,
            'anomaly_risk': anomaly_risk,
            'ml_risk': ml_risk
        }

        # Recalculate total risk with ML components
        weights = {
            'volatility_risk': 0.2,
            'liquidity_risk': 0.2,
            'spread_risk': 0.15,
            'volume_risk': 0.15,
            'anomaly_risk': 0.15,
            'ml_risk': 0.15
        }

        enhanced_metrics['total_risk'] = sum(enhanced_metrics[k] * weights[k] 
                                           for k in weights.keys())

        return enhanced_metrics

    def get_trade_recommendations(self, risk_metrics, trade_size):
        """Enhanced trade recommendations"""
        recommendations = super().get_trade_recommendations(risk_metrics, trade_size)
        
        # Add ML-based insights
        if risk_metrics['anomaly_risk'] > 0.3:
            recommendations['warnings'] = ["Unusual market behavior detected"]
        
        if risk_metrics['ml_risk'] > 0.7:
            recommendations['warnings'].append("High risk predicted by ML model")
        
        return recommendations

    def analyze_market_sentiment(self, symbol):
        """Analyze market sentiment using NLP"""
        try:
            # Fetch news headlines (implement your news API here)
            news_headlines = []  # Replace with actual news API call
            
            sentiments = []
            for headline in news_headlines:
                inputs = self.sentiment_tokenizer(headline, return_tensors="pt")
                outputs = self.sentiment_model(**inputs)
                sentiment = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sentiments.append(sentiment.detach().numpy()[0])
            
            return np.mean(sentiments, axis=0)
        except:
            return None
