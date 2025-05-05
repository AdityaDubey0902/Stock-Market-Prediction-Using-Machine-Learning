
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load the data
df = pd.read_csv('Bharti Airtel Stock Price History.csv')

# Convert 'Date' to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Convert string columns to numeric (remove commas and % signs)
numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
for col in numeric_cols:
    # Remove commas and '%' signs
    df[col] = df[col].str.replace(',', '', regex=True).str.replace('%', '', regex=True)

    # Handle 'M' (million) and 'K' (thousands) in 'Vol.' column
    if col == 'Vol.':
        # Define a function to convert volume string to float
        def convert_volume(volume_str):
            if 'M' in volume_str:
                return float(volume_str.replace('M', '')) * 1000000
            elif 'K' in volume_str:
                return float(volume_str.replace('K', '')) * 1000
            else:
                return float(volume_str)

        df[col] = df[col].apply(convert_volume) # Apply the function to the 'Vol.' column
    else:
        df[col] = df[col].astype(float)

# Rename 'Price' to 'Close' for candlestick chart
df = df.rename(columns={'Price': 'Close'})

# 1. Candlestick Chart
plt.figure(figsize=(15, 7))

# Prepare data for candlestick chart
df_reset = df.reset_index()
df_reset['Date'] = df_reset['Date'].map(mdates.date2num)

# Create subplot for candlestick chart
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=4, colspan=1)
plt.title('Reliance Industries Stock Price Candlestick Chart')

# Plot candlestick
candlestick_ohlc(ax1, df_reset[['Date', 'Open', 'High', 'Low', 'Close']].values, 
                 width=0.6, colorup='g', colordown='r')
ax1.xaxis_date()
ax1.grid(True)

# Create subplot for volume
ax2 = plt.subplot2grid((6,1), (4,0), rowspan=2, colspan=1, sharex=ax1)
ax2.bar(df_reset['Date'], df_reset['Vol.'], width=0.6, align='center')
ax2.grid(True)
plt.ylabel('Volume')

# Format x-axis
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

# 2. Closing Price Graph
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
plt.title('Reliance Industries Closing Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Stock Features')
plt.tight_layout()
plt.show()

# Prepare data for LSTM model
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length), 0])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(scaled_data, seq_length)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

train_mae, train_rmse, train_r2 = calculate_metrics(y_train_actual, train_predict)
test_mae, test_rmse, test_r2 = calculate_metrics(y_test_actual, test_predict)

# Prepare data for traditional ML models
X_train_ml = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test_ml = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'XGBoost': XGBRegressor(n_estimators=100, verbosity=0)
}

results = {}

# Train and evaluate each model
for name, model_ml in models.items():
    model_ml.fit(X_train_ml, y_train)
    preds = model_ml.predict(X_test_ml)
    
    # Rescale predictions
    preds_scaled = scaler.inverse_transform(preds.reshape(-1, 1))
    actual_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae, rmse, r2 = calculate_metrics(actual_scaled, preds_scaled)
    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Predictions': preds_scaled
    }

# Add LSTM results
results['LSTM'] = {
    'MAE': test_mae,
    'RMSE': test_rmse,
    'R2': test_r2,
    'Predictions': test_predict
}

# 4. Evaluation Metrics Comparison Graph
metrics = ['MAE', 'RMSE', 'R2']
model_names = list(results.keys())

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot each metric
for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in model_names]
    if metric == 'R2':
        # For R2, we might want to show higher values are better
        bars = axes[i].bar(model_names, values, color='skyblue')
        axes[i].set_title(f'{metric} Comparison (Higher is better)')
    else:
        bars = axes[i].bar(model_names, values, color='salmon')
        axes[i].set_title(f'{metric} Comparison (Lower is better)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)

plt.suptitle('Model Performance Comparison Across Different Metrics', y=1.05)
plt.tight_layout()
plt.show()

# Create comparison table
comparison_data = {
    'Model': model_names,
    'MAE': [results[m]['MAE'] for m in model_names],
    'RMSE': [results[m]['RMSE'] for m in model_names],
    'R2': [results[m]['R2'] for m in model_names]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Evaluation Comparison Table:")
print(comparison_df.sort_values(by='RMSE'))