import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradeRiskModel:
    def __init__(self):
        """Initialize the model with necessary components"""
        # Scalers
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        
        # ML Models
        self.anomaly_detector = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        self.xgb_model = None
        self.lstm_model = None
        self.rf_model = None
        
        # Technical Indicators
        self.technical_indicators = {}
        
        # Configure for CPU-only
        tf.config.set_visible_devices([], 'GPU')

    def initialize_models(self):
        """Initialize and configure ML models"""
        try:
            # XGBoost model
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            # Random Forest model
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            
            # LSTM model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 6)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            self.lstm_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
        except Exception as e:
            raise Exception(f"Error initializing models: {str(e)}")

    def is_market_open(self, exchange="US"):
        """Check if the market is currently open"""
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

    def fetch_market_data(self, symbol, period='1d', interval='1m'):
        """Fetch and validate market data"""
        try:
            is_indian = symbol.endswith('.NS')
            exchange = "India" if is_indian else "US"
            
            # Adjust period and interval based on market hours
            if period == '1d':
                interval = '5m'  # Use 5-minute intervals for better data availability
            elif period in ['5d', '1mo']:
                interval = '15m'
            else:
                interval = '1h'
    
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                # Try fetching with different parameters
                data = ticker.history(period='1mo', interval='1h')
                if data.empty:
                    raise ValueError(f"No data available for {symbol}. Please verify the symbol.")
            
            # Process the data
            processed_data = self.process_market_data(data)
            
            if len(processed_data) < 2:
                raise ValueError(f"Insufficient data for {symbol}")
                
            return processed_data
            
        except Exception as e:
            if "Symbol may be delisted" in str(e):
                raise Exception(f"Symbol {symbol} may be delisted or invalid")
            elif "401" in str(e):
                raise Exception(f"API access error. Please try again later")
            else:
                raise Exception(f"Error fetching data for {symbol}: {str(e)}")

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        try:
            # Moving Averages
            sma = SMAIndicator(data['Close'], window=20)
            ema = EMAIndicator(data['Close'], window=20)
            
            # Bollinger Bands
            bb = BollingerBands(data['Close'])
            
            # RSI
            rsi = RSIIndicator(data['Close'])
            
            indicators = pd.DataFrame({
                'SMA': sma.sma_indicator(),
                'EMA': ema.ema_indicator(),
                'BB_upper': bb.bollinger_hband(),
                'BB_lower': bb.bollinger_lband(),
                'RSI': rsi.rsi()
            })
            
            return indicators
            
        except Exception as e:
            raise Exception(f"Error calculating technical indicators: {str(e)}")

    def process_market_data(self, data):
        """Process and enrich market data"""
        try:
            if data.empty:
                raise ValueError("No market data available to process")
                
            # Calculate basic metrics
            data['Volatility'] = data['Close'].rolling(window=20).std()
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Price_Change'] = data['Close'].pct_change()
            data['Spread'] = data['High'] - data['Low']
            
            # Add technical indicators
            indicators = self.calculate_technical_indicators(data)
            data = pd.concat([data, indicators], axis=1)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error processing market data: {str(e)}")

    def prepare_ml_features(self, data):
        """Prepare features for ML models"""
        try:
            features = [
                'Close', 'Volume', 'Volatility', 'Volume_MA', 
                'Price_Change', 'Spread', 'SMA', 'EMA', 'RSI'
            ]
            
            X = data[features].fillna(method='ffill')
            X = self.scaler.fit_transform(X)
            
            return X
            
        except Exception as e:
            raise Exception(f"Error preparing ML features: {str(e)}")

    def detect_anomalies(self, data):
        """Detect market anomalies"""
        try:
            X = self.prepare_ml_features(data)
            anomalies = self.anomaly_detector.fit_predict(X)
            return anomalies
            
        except Exception as e:
            raise Exception(f"Error detecting anomalies: {str(e)}")

    def predict_price_movement(self, data):
        """Predict future price movements"""
        try:
            X = self.prepare_ml_features(data)
            
            # Ensemble predictions
            xgb_pred = self.xgb_model.predict(X[-1:]) if self.xgb_model else 0
            rf_pred = self.rf_model.predict(X[-1:]) if self.rf_model else 0
            
            # Combine predictions
            ensemble_pred = (xgb_pred + rf_pred) / 2
            
            return ensemble_pred[0]
            
        except Exception as e:
            raise Exception(f"Error predicting price movement: {str(e)}")

    def calculate_risk_metrics(self, market_data, trade_size):
        """Calculate comprehensive risk metrics"""
        try:
            if market_data.empty:
                raise ValueError("No market data available")
                
            latest_data = market_data.iloc[-1]
            avg_volume = market_data['Volume'].mean()
            
            # Basic risk metrics
            risk_metrics = {
                'volatility_risk': min(1, latest_data['Volatility'] / latest_data['Close'] if latest_data['Close'] != 0 else 1),
                'liquidity_risk': min(1, trade_size / avg_volume if avg_volume != 0 else 1),
                'spread_risk': min(1, latest_data['Spread'] / latest_data['Close'] if latest_data['Close'] != 0 else 1),
                'volume_risk': min(1, 1 - (latest_data['Volume'] / latest_data['Volume_MA'] if latest_data['Volume_MA'] != 0 else 1))
            }
            
            # ML-based risk metrics
            anomaly_score = np.mean(self.detect_anomalies(market_data) == -1)
            price_movement = self.predict_price_movement(market_data)
            
            risk_metrics.update({
                'anomaly_risk': anomaly_score,
                'price_movement_risk': max(0, min(1, abs(price_movement)))
            })
            
            # Calculate total risk
            weights = {
                'volatility_risk': 0.2,
                'liquidity_risk': 0.2,
                'spread_risk': 0.15,
                'volume_risk': 0.15,
                'anomaly_risk': 0.15,
                'price_movement_risk': 0.15
            }
            
            total_risk = sum(risk_metrics[k] * weights[k] for k in weights.keys())
            risk_metrics['total_risk'] = min(1, total_risk)
            
            return risk_metrics
            
        except Exception as e:
            raise Exception(f"Error calculating risk metrics: {str(e)}")

    def get_trade_recommendations(self, risk_metrics, trade_size):
        """Generate trading recommendations"""
        try:
            total_risk = risk_metrics['total_risk']
            
            if total_risk > 0.8:
                risk_level = "High"
                action = "Avoid trading or reduce position size significantly"
                suggested_size = trade_size * 0.3
            elif total_risk > 0.6:
                risk_level = "Medium-High"
                action = "Consider splitting trade or waiting for better conditions"
                suggested_size = trade_size * 0.5
            elif total_risk > 0.4:
                risk_level = "Medium"
                action = "Proceed with caution, consider reducing position size"
                suggested_size = trade_size * 0.75
            else:
                risk_level = "Low"
                action = "Proceed with trade as planned"
                suggested_size = trade_size
                
            return {
                'risk_level': risk_level,
                'action': action,
                'suggested_size': int(suggested_size),
                'original_size': trade_size,
                'risk_score': total_risk,
                'confidence': 1 - total_risk
            }
            
        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")

    def get_optimal_execution_time(self, market_data):
        """Determine optimal execution time"""
        try:
            if market_data.empty:
                raise ValueError("No market data available")
                
            market_data['Hour'] = pd.to_datetime(market_data.index).hour
            
            # Analyze hourly patterns
            hourly_metrics = market_data.groupby('Hour').agg({
                'Volatility': 'mean',
                'Volume': 'mean',
                'Spread': 'mean',
                'RSI': 'mean'
            })
            
            # Score each hour
            hourly_metrics = (hourly_metrics - hourly_metrics.mean()) / hourly_metrics.std()
            best_hour = hourly_metrics.mean(axis=1).idxmin()
            
            current_hour = datetime.now().hour
            
            if best_hour > current_hour:
                return f"Optimal execution time: Today at {best_hour:02d}:00"
            else:
                return f"Optimal execution time: Tomorrow at {best_hour:02d}:00"
                
        except Exception as e:
            raise Exception(f"Error determining optimal execution time: {str(e)}")
