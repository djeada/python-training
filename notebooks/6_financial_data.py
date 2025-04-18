## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## 2. Fetch and Display Financial Data
# Fetch historical stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
data['Daily Return'] = data['Close'].pct_change()
data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()

display(data.head())

## 3. Data Visualization
# Plot stock prices and cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Cumulative Return'], label='Cumulative Return')
plt.title(f'{ticker} Stock Price and Cumulative Return')
plt.legend()
plt.show()

## 4. Moving Averages
# Calculate moving averages
data['20-day MA'] = data['Close'].rolling(window=20).mean()
data['50-day MA'] = data['Close'].rolling(window=50).mean()

# Plot moving averages
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['20-day MA'], label='20-Day MA', linestyle='--')
plt.plot(data['50-day MA'], label='50-Day MA', linestyle='--')
plt.title(f'{ticker} Moving Averages')
plt.legend()
plt.show()

## 5. Risk and Portfolio Management
# Calculate portfolio risk and return
weights = np.array([0.5, 0.5])  # Example: 50% AAPL, 50% MSFT
returns = data['Daily Return']
portfolio_return = np.dot(weights, returns.mean()) * 252  # Annualized return
portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized risk
sharpe_ratio = portfolio_return / portfolio_std_dev

print(f'Portfolio Return: {portfolio_return:.2f}')
print(f'Portfolio Risk: {portfolio_std_dev:.2f}')
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

## 6. ARIMA Time Series Forecasting
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(data['Close'].dropna(), order=(5, 1, 0))  # ARIMA(5,1,0)
result = model.fit()

# Forecast future prices
forecast = result.forecast(steps=30)
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Historical Prices')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.title(f'{ticker} ARIMA Forecast')
plt.legend()
plt.show()

## 7. Trading Strategy and Backtesting
# Simple momentum strategy
data['Signal'] = np.where(data['20-day MA'] > data['50-day MA'], 1, 0)
data['Strategy Return'] = data['Signal'].shift(1) * data['Daily Return']

# Backtest performance
cumulative_strategy_return = (1 + data['Strategy Return']).cumprod()
plt.figure(figsize=(12, 6))
plt.plot(data['Cumulative Return'], label='Buy-and-Hold')
plt.plot(cumulative_strategy_return, label='Momentum Strategy')
plt.title(f'{ticker} Strategy Backtest')
plt.legend()
plt.show()

## 8. Machine Learning for Prediction
# Prepare data for ML
data['Target'] = np.where(data['Daily Return'] > 0, 1, 0)
features = ['20-day MA', '50-day MA']
data.dropna(inplace=True)
X = data[features]
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Prediction Accuracy: {accuracy:.2f}')
