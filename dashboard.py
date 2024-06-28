import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
file_path = 'C:/Users/satyendra maurya/.ipynb_checkpoints/nvidia_stock_2015_to_2024.csv'
nvidia_stock = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
nvidia_stock['date'] = pd.to_datetime(nvidia_stock['date'])
nvidia_stock.set_index('date', inplace=True)

# Streamlit App
st.title('NVIDIA Corp. Stock 2015-2024 Analysis Dashboard')

# Display the dataset
st.subheader('Raw Data')
st.write(nvidia_stock.head())

# Basic Information and Summary Statistics
st.subheader('Basic Information')
st.write(nvidia_stock.info())
st.write(nvidia_stock.describe())

# Missing Values
st.subheader('Missing Values')
st.write(nvidia_stock.isnull().sum())

# Exploratory Data Analysis (EDA)
st.subheader('Exploratory Data Analysis (EDA)')

# Plot the stock price over time
st.write('### Stock Price Over Time')
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(nvidia_stock.index, nvidia_stock['close'], label='Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('NVIDIA Stock Price Over Time')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Distribution of the Stock Prices
st.write('### Distribution of Close Prices')
fig, ax = plt.subplots(figsize=(12, 8))
sns.histplot(nvidia_stock['close'], kde=True, ax=ax)
ax.set_title('Distribution of NVIDIA Close Prices')
ax.set_xlabel('Close Price')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Moving Averages
nvidia_stock['MA20'] = nvidia_stock['close'].rolling(window=20).mean()
nvidia_stock['MA50'] = nvidia_stock['close'].rolling(window=50).mean()

st.write('### Stock Price with Moving Averages')
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(nvidia_stock.index, nvidia_stock['close'], label='Close Price')
ax.plot(nvidia_stock.index, nvidia_stock['MA20'], label='20-Day MA')
ax.plot(nvidia_stock.index, nvidia_stock['MA50'], label='50-Day MA')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('NVIDIA Stock Price with Moving Averages')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Volatility Analysis
st.write('### Volatility Analysis')
nvidia_stock['Volatility20'] = nvidia_stock['close'].rolling(window=20).std()
nvidia_stock['Volatility30'] = nvidia_stock['close'].rolling(window=30).std()

fig, ax = plt.subplots(figsize=(17, 7))
ax.plot(nvidia_stock.index, nvidia_stock['Volatility20'], label='20-Day Volatility', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility (20-Day Rolling Std)')
ax.set_title('NVIDIA Stock Volatility Over Time')
ax.legend()
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(17, 7))
ax.plot(nvidia_stock.index, nvidia_stock['Volatility30'], label='30-Day Volatility', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility (30-Day Rolling Std)')
ax.set_title('NVIDIA Stock Volatility Over Time')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Correlation Analysis
st.write('### Correlation Matrix')
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = nvidia_stock.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix')
st.pyplot(fig)

# Candlestick Chart
st.write('### Candlestick Chart')
fig = go.Figure(data=[go.Candlestick(x=nvidia_stock.index,
                                     open=nvidia_stock['open'],
                                     high=nvidia_stock['high'],
                                     low=nvidia_stock['low'],
                                     close=nvidia_stock['close'])])
fig.update_layout(title='NVIDIA Candlestick Chart',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)
st.plotly_chart(fig)

# Daily Returns Analysis
st.write('### Daily Returns Analysis')
nvidia_stock['Daily Return'] = nvidia_stock['close'].pct_change()

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(nvidia_stock.index, nvidia_stock['Daily Return'])
ax.set_xlabel('Date')
ax.set_ylabel('Daily Return')
ax.set_title('Daily Returns of NVIDIA Stock')
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(nvidia_stock['Daily Return'].dropna(), kde=True, bins=50, ax=ax)
ax.set_title('Distribution of Daily Returns')
ax.set_xlabel('Daily Return')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Monthly Returns Analysis
st.write('### Monthly Returns Analysis')
monthly_returns = nvidia_stock['close'].resample('M').ffill().pct_change()

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(monthly_returns.index, monthly_returns)
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Return')
ax.set_title('Monthly Returns of NVIDIA Stock')
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(monthly_returns.dropna(), kde=True, bins=50, ax=ax)
ax.set_title('Distribution of Monthly Returns')
ax.set_xlabel('Monthly Return')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Annual Returns Analysis
st.write('### Annual Returns Analysis')
annual_returns = nvidia_stock['close'].resample('Y').ffill().pct_change()

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(annual_returns.index, annual_returns, marker='o')
ax.set_xlabel('Date')
ax.set_ylabel('Annual Return')
ax.set_title('Annual Returns of NVIDIA Stock')
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(annual_returns.dropna(), kde=True, bins=10, ax=ax)
ax.set_title('Distribution of Annual Returns')
ax.set_xlabel('Annual Return')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Rolling Statistics
st.write('### Rolling Statistics')
window = 20
nvidia_stock['Rolling Mean'] = nvidia_stock['close'].rolling(window=window).mean()
nvidia_stock['Rolling Std'] = nvidia_stock['close'].rolling(window=window).std()

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(nvidia_stock.index, nvidia_stock['close'], label='Close Price')
ax.plot(nvidia_stock.index, nvidia_stock['Rolling Mean'], label=f'{window}-Day Rolling Mean')
ax.plot(nvidia_stock.index, nvidia_stock['Rolling Std'], label=f'{window}-Day Rolling Std', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price / Rolling Statistics')
ax.set_title('Rolling Statistics')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Bollinger Bands
st.write('### Bollinger Bands')
nvidia_stock['Bollinger Upper'] = nvidia_stock['Rolling Mean'] + (nvidia_stock['Rolling Std'] * 2)
nvidia_stock['Bollinger Lower'] = nvidia_stock['Rolling Mean'] - (nvidia_stock['Rolling Std'] * 2)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(nvidia_stock.index, nvidia_stock['close'], label='Close Price')
ax.plot(nvidia_stock.index, nvidia_stock['Rolling Mean'], label='Rolling Mean')
ax.plot(nvidia_stock.index, nvidia_stock['Bollinger Upper'], label='Bollinger Upper', linestyle='--', color='red')
ax.plot(nvidia_stock.index, nvidia_stock['Bollinger Lower'], label='Bollinger Lower', linestyle='--', color='blue')
ax.fill_between(nvidia_stock.index, nvidia_stock['Bollinger Lower'], nvidia_stock['Bollinger Upper'], alpha=0.1)
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('Bollinger Bands')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Relative Strength Index (RSI)
st.write('### Relative Strength Index (RSI)')
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

nvidia_stock['RSI'] = calculate_rsi(nvidia_stock['close'])

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(nvidia_stock.index, nvidia_stock['RSI'], label='RSI', color='purple')
ax.axhline(30, linestyle='--', alpha=0.5, color='red')
ax.axhline(70, linestyle='--', alpha=0.5, color='green')
ax.set_xlabel('Date')
ax.set_ylabel('RSI')
ax.set_title('Relative Strength Index (RSI)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Autocorrelation and Partial Autocorrelation Plots
st.write('### Autocorrelation and Partial Autocorrelation Plots')
fig, ax = plt.subplots(2, 1, figsize=(12, 12))

plot_acf(nvidia_stock['close'], lags=50, ax=ax[0])
ax[0].set_title('Autocorrelation Function')

plot_pacf(nvidia_stock['close'], lags=50, ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function')

st.pyplot(fig)

# Seasonal Decomposition
st.write('### Seasonal Decomposition')
decomposition = seasonal_decompose(nvidia_stock['close'], model='multiplicative', period=365)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))
decomposition.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
decomposition.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
st.pyplot(fig)

# ARIMA Model
st.write('### ARIMA Model Forecasting')
train, test = train_test_split(nvidia_stock['close'], test_size=0.2, shuffle=False)
arima_model = ARIMA(train, order=(5, 1, 0))
arima_result = arima_model.fit()

arima_forecast = arima_result.forecast(steps=len(test))
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(train.index, train, label='Train')
ax.plot(test.index, test, label='Test')
ax.plot(test.index, arima_forecast, label='ARIMA Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('ARIMA Model Forecast')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.write(f'ARIMA Model MSE: {mean_squared_error(test, arima_forecast):.4f}')

# LSTM Model
st.write('### LSTM Model Forecasting')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(nvidia_stock['close'].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=10)

lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

index_for_plotting = nvidia_stock.index[-len(lstm_predictions):]


fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(nvidia_stock.index[-len(test_data):], nvidia_stock['close'].iloc[-len(test_data):], label='True Price')
ax.plot(index_for_plotting, lstm_predictions, label='LSTM Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('LSTM Model Forecast')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Correctly match the lengths for MSE calculation
true_values_for_mse = nvidia_stock["close"].iloc[-len(lstm_predictions):]
st.write(f'LSTM Model MSE: {mean_squared_error(true_values_for_mse, lstm_predictions):.4f}')
