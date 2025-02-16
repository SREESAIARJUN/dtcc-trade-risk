import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class TradeRiskModel:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def is_market_open(self, exchange="US"):
        now = datetime.now()
        if exchange == "India":
            # Indian Market Hours (9:15 AM - 3:30 PM IST)
            india_time = now + timedelta(hours=5, minutes=30)  # Convert to IST
            market_open = india_time.replace(hour=9, minute=15)
            market_close = india_time.replace(hour=15, minute=30)
            return market_open <= india_time <= market_close
        else:
            # US Market Hours (9:30 AM - 4:00 PM EST)
            est_time = now - timedelta(hours=5)  # Convert to EST
            market_open = est_time.replace(hour=9, minute=30)
            market_close = est_time.replace(hour=16, minute=0)
            return market_open <= est_time <= market_close

    def convert_to_usd(self, price, from_currency="INR"):
        if from_currency == "INR":
            # Approximate USD/INR rate (you might want to fetch real-time rates)
            usd_inr_rate = 83.0
            return price / usd_inr_rate
        return price

    def fetch_market_data(self, symbol, period='1d', interval='1m'):
        """Fetch real-time market data using yfinance"""
        try:
            # Determine exchange
            is_indian = symbol.endswith('.NS')
            exchange = "India" if is_indian else "US"
            
            # Check market hours
            if not self.is_market_open(exchange):
                market_hours = "9:15 AM - 3:30 PM IST" if is_indian else "9:30 AM - 4:00 PM EST"
                raise ValueError(f"{exchange} markets are closed (Trading hours: {market_hours})")

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                if is_indian:
                    raise ValueError(f"No data available for {symbol}. Please verify the symbol.")
                else:
                    raise ValueError(f"No data available for symbol {symbol}")
            
            # Convert prices to USD if Indian stock
            if is_indian:
                for col in ['Open', 'High', 'Low', 'Close']:
                    data[col] = data[col].apply(lambda x: self.convert_to_usd(x, "INR"))
                
            return self.process_market_data(data)
        except Exception as e:
            if "not found" in str(e).lower():
                suggestion = f"{symbol[:-3]} (without .NS)" if is_indian else f"{symbol}.NS"
                raise ValueError(f"Symbol {symbol} not found. Did you mean {suggestion}?")
            raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

    def process_market_data(self, data):
        """Calculate market metrics"""
        if data.empty:
            raise ValueError("No market data available to process")
            
        data['Volatility'] = data['Close'].rolling(window=20).std()
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Price_Change'] = data['Close'].pct_change()
        data['Spread'] = data['High'] - data['Low']
        return data
    
    def calculate_risk_metrics(self, market_data, trade_size):
        """Calculate comprehensive risk metrics"""
        if market_data.empty:
            return {
                'volatility_risk': 1.0,
                'liquidity_risk': 1.0,
                'spread_risk': 1.0,
                'volume_risk': 1.0,
                'total_risk': 1.0
            }
            
        latest_data = market_data.iloc[-1]
        avg_volume = market_data['Volume'].mean()
        
        risk_metrics = {
            'volatility_risk': min(1, latest_data['Volatility'] / latest_data['Close'] if latest_data['Close'] != 0 else 1),
            'liquidity_risk': min(1, trade_size / avg_volume if avg_volume != 0 else 1),
            'spread_risk': min(1, latest_data['Spread'] / latest_data['Close'] if latest_data['Close'] != 0 else 1),
            'volume_risk': min(1, 1 - (latest_data['Volume'] / latest_data['Volume_MA'] if latest_data['Volume_MA'] != 0 else 1))
        }
        
        weights = {
            'volatility_risk': 0.3,
            'liquidity_risk': 0.3,
            'spread_risk': 0.2,
            'volume_risk': 0.2
        }
        
        total_risk = sum(risk_metrics[k] * weights[k] for k in risk_metrics)
        risk_metrics['total_risk'] = min(1, total_risk)
        
        return risk_metrics
    
    def get_trade_recommendations(self, risk_metrics, trade_size):
        """Generate trade recommendations based on risk metrics"""
        total_risk = risk_metrics['total_risk']
        
        if total_risk > 0.8:
            risk_level = "High"
            action = "Avoid trading or split into smaller orders"
            suggested_size = trade_size * 0.5
        elif total_risk > 0.5:
            risk_level = "Medium"
            action = "Consider splitting trade or waiting for better conditions"
            suggested_size = trade_size * 0.75
        else:
            risk_level = "Low"
            action = "Proceed with trade"
            suggested_size = trade_size
            
        return {
            'risk_level': risk_level,
            'action': action,
            'suggested_size': int(suggested_size),
            'original_size': trade_size
        }
    
    def get_optimal_execution_time(self, market_data):
        """Suggest optimal execution time based on historical patterns"""
        if market_data.empty:
            return "Unable to determine optimal execution time due to lack of data"
            
        market_data['Hour'] = pd.to_datetime(market_data.index).hour
        
        hourly_risk = market_data.groupby('Hour').agg({
            'Volatility': 'mean',
            'Volume': 'mean',
            'Spread': 'mean'
        })
        
        hourly_risk = (hourly_risk - hourly_risk.mean()) / hourly_risk.std()
        best_hour = hourly_risk.mean(axis=1).idxmin()
        
        current_hour = datetime.now().hour
        
        if best_hour > current_hour:
            return f"Suggest waiting until {best_hour}:00"
        else:
            return f"Suggest waiting until tomorrow at {best_hour}:00"
