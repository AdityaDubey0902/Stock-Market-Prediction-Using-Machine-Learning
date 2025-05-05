import streamlit as st
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

# Page configuration
st.set_page_config(page_title="Stock Price Analysis", layout="wide")

# Title
st.title("Stock Price Analysis & Prediction")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Bharti Airtel Stock Price History.csv')
        
        # Convert 'Date' to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Convert string columns to numeric
        numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
        for col in numeric_cols:
            if col in df.columns:
                # Remove commas and '%' signs
                df[col] = df[col].astype(str).str.replace(',', '', regex=True).str.replace('%', '', regex=True)

                if col == 'Vol.':
                    def convert_volume(volume_str):
                        if 'M' in volume_str:
                            return float(volume_str.replace('M', '')) * 1000000
                        elif 'K' in volume_str:
                            return float(volume_str.replace('K', '')) * 1000
                        else:
                            return float(volume_str)
                    df[col] = df[col].apply(convert_volume)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.rename(columns={'Price': 'Close'})
        return df.dropna()  # Remove any rows with missing values
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Check if data loaded successfully
if df.empty:
    st.error("No data available. Please check your data file.")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
show_raw_data = st.sidebar.checkbox("Show raw data")

# Set default dates based on available data
min_date = df.index.min()
max_date = df.index.max()

start_date = st.sidebar.date_input(
    "Start date", 
    value=min_date,
    min_value=min_date,
    max_value=max_date
)
end_date = st.sidebar.date_input(
    "End date", 
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# Validate date range
if start_date > end_date:
    st.sidebar.error("End date must be after start date")
    st.stop()

# Filter data based on date selection
filtered_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

# If filtered data is empty, use the full dataset with warning
if filtered_df.empty:
    # st.warning("No data available for selected date range. Showing full dataset instead.")
    filtered_df = df.copy()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Charts", "Correlation", "Model Performance", "Predictions"])

with tab1:
    # Candlestick chart
    st.subheader("Candlestick Chart with Volume")
    
    if not filtered_df.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        df_reset = filtered_df.reset_index()
        df_reset['Date'] = df_reset['Date'].map(mdates.date2num)
        
        candlestick_ohlc(ax1, df_reset[['Date', 'Open', 'High', 'Low', 'Close']].values, 
                        width=0.6, colorup='g', colordown='r')
        ax1.xaxis_date()
        ax1.grid(True)
        ax1.set_title('Price Movement')
        
        ax2.bar(df_reset['Date'], df_reset['Vol.'], width=0.6, align='center')
        ax2.grid(True)
        ax2.set_ylabel('Volume')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data available for visualization")

    # Closing price line chart
    st.subheader("Closing Price Trend")
    if not filtered_df.empty and 'Close' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(filtered_df.index, filtered_df['Close'], label='Closing Price', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("No closing price data available")

with tab2:
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    if not filtered_df.empty and len(filtered_df.columns) > 1:
        numeric_df = filtered_df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation analysis")
    else:
        st.warning("No data available for correlation analysis")

with tab3:
    # Model training and evaluation
    st.subheader("Model Training and Evaluation")
    
    if st.button("Train Models and Evaluate Performance"):
        if len(filtered_df) < 20:  # Minimum data points needed
            st.error(f"Not enough data points ({len(filtered_df)}). Please select a wider date range with at least 20 data points.")
        elif 'Close' not in filtered_df.columns:
            st.error("No closing price data available for modeling.")
        else:
            with st.spinner("Training models... This may take a few minutes"):
                try:
                    # Prepare data for modeling
                    data = filtered_df['Close'].values.reshape(-1, 1)
                    
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(data)
                    
                    # Create sequences
                    def create_sequences(data, seq_length):
                        X, y = [], []
                        for i in range(len(data)-seq_length-1):
                            X.append(data[i:(i+seq_length), 0])
                            y.append(data[i+seq_length, 0])
                        return np.array(X), np.array(y)
                    
                    seq_length = min(5, len(scaled_data) - 2)  # Ensure we have enough data
                    X, y = create_sequences(scaled_data, seq_length)
                    
                    # Split data
                    train_size = max(1, int(len(X) * 0.8))  # Ensure at least 1 sample
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Reshape for LSTM
                    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    
                    # Train LSTM model
                    lstm_model = Sequential()
                    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
                    lstm_model.add(Dropout(0.2))
                    lstm_model.add(LSTM(50, return_sequences=True))
                    lstm_model.add(Dropout(0.2))
                    lstm_model.add(LSTM(50))
                    lstm_model.add(Dropout(0.2))
                    lstm_model.add(Dense(1))
                    
                    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
                    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
                    
                    # Make predictions
                    lstm_train_pred = lstm_model.predict(X_train_lstm)
                    lstm_test_pred = lstm_model.predict(X_test_lstm)
                    
                    # Inverse transform
                    lstm_train_pred = scaler.inverse_transform(lstm_train_pred)
                    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                    lstm_test_pred = scaler.inverse_transform(lstm_test_pred)
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate metrics
                    def calculate_metrics(y_true, y_pred):
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        r2 = r2_score(y_true, y_pred)
                        return mae, rmse, r2
                    
                    lstm_train_mae, lstm_train_rmse, lstm_train_r2 = calculate_metrics(y_train_actual, lstm_train_pred)
                    lstm_test_mae, lstm_test_rmse, lstm_test_r2 = calculate_metrics(y_test_actual, lstm_test_pred)
                    
                    # Prepare data for traditional ML models
                    X_train_ml = X_train.reshape(X_train.shape[0], X_train.shape[1])
                    X_test_ml = X_test.reshape(X_test.shape[0], X_test.shape[1])
                    
                    # Define models
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Decision Tree': DecisionTreeRegressor(random_state=42),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                        'XGBoost': XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
                    }
                    
                    results = {}
                    
                    # Train and evaluate each model
                    for name, model in models.items():
                        model.fit(X_train_ml, y_train)
                        preds = model.predict(X_test_ml)
                        preds_scaled = scaler.inverse_transform(preds.reshape(-1, 1))
                        mae, rmse, r2 = calculate_metrics(y_test_actual, preds_scaled)
                        results[name] = {
                            'MAE': mae,
                            'RMSE': rmse,
                            'R2': r2,
                            'Predictions': preds_scaled
                        }
                    
                    # Add LSTM results
                    results['LSTM'] = {
                        'MAE': lstm_test_mae,
                        'RMSE': lstm_test_rmse,
                        'R2': lstm_test_r2,
                        'Predictions': lstm_test_pred
                    }
                    
                    # Display metrics comparison
                    st.subheader("Model Performance Comparison")
                    
                    # Metrics comparison chart
                    metrics = ['MAE', 'RMSE', 'R2']
                    model_names = list(results.keys())
                    
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    
                    for i, metric in enumerate(metrics):
                        values = [results[model][metric] for model in model_names]
                        if metric == 'R2':
                            bars = axes[i].bar(model_names, values, color='skyblue')
                            axes[i].set_title(f'{metric} (Higher is better)')
                        else:
                            bars = axes[i].bar(model_names, values, color='salmon')
                            axes[i].set_title(f'{metric} (Lower is better)')
                        
                        for bar in bars:
                            height = bar.get_height()
                            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                        f'{height:.2f}',
                                        ha='center', va='bottom')
                        
                        axes[i].set_ylabel(metric)
                        axes[i].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Metrics table
                    st.subheader("Detailed Metrics Table")
                    comparison_data = {
                        'Model': model_names,
                        'MAE': [results[m]['MAE'] for m in model_names],
                        'RMSE': [results[m]['RMSE'] for m in model_names],
                        'R2': [results[m]['R2'] for m in model_names]
                    }
                    
                    st.dataframe(pd.DataFrame(comparison_data).sort_values(by='RMSE'))
                    
                    # Actual vs Predicted plot
                    st.subheader("Actual vs Predicted Prices (Test Set)")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(y_test_actual, label='Actual Prices', color='blue')
                    
                    # Plot predictions from each model
                    colors = ['green', 'orange', 'purple', 'brown', 'red']
                    for i, (name, result) in enumerate(results.items()):
                        ax.plot(result['Predictions'], label=f'{name} Predictions', 
                               color=colors[i], alpha=0.7, linestyle='--')
                    
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Price')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"An error occurred during model training: {str(e)}")
                    st.stop()

with tab4:
    st.subheader("Future Price Prediction")
    st.info("""
    This feature would require implementing a forecasting function that predicts future prices based on the trained models.
    To implement this, we would need to:
    1. Save the trained models
    2. Create a forecasting function that generates future sequences
    3. Visualize the predictions with confidence intervals
    """)
    st.write("Feature coming soon...")

# Show raw data if selected
if show_raw_data and not filtered_df.empty:
    st.subheader("Raw Data")
    st.dataframe(filtered_df)
elif show_raw_data:
    st.warning("No data available to display")

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from mplfinance.original_flavor import candlestick_ohlc
# # import matplotlib.dates as mdates
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from keras.models import Sequential
# # from keras.layers import LSTM, Dense, Dropout
# # from keras.callbacks import EarlyStopping
# # from sklearn.linear_model import LinearRegression
# # from sklearn.tree import DecisionTreeRegressor
# # from sklearn.ensemble import RandomForestRegressor
# # from xgboost import XGBRegressor

# # # Page configuration
# # st.set_page_config(page_title="Stock Price Analysis", layout="wide")

# # # Title
# # st.title("Reliance Industries Stock Price Analysis & Prediction")

# # # Load data
# # @st.cache_data
# # def load_data():
# #     try:
# #         df = pd.read_csv('Reliance Industries Stock Price History.csv')
        
# #         # Convert 'Date' to datetime and set as index
# #         df['Date'] = pd.to_datetime(df['Date'])
# #         df.set_index('Date', inplace=True)

# #         # Convert string columns to numeric
# #         numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
# #         for col in numeric_cols:
# #             if col in df.columns:
# #                 # Remove commas and '%' signs
# #                 df[col] = df[col].astype(str).str.replace(',', '', regex=True).str.replace('%', '', regex=True)

# #                 if col == 'Vol.':
# #                     def convert_volume(volume_str):
# #                         if 'M' in volume_str:
# #                             return float(volume_str.replace('M', '')) * 1000000
# #                         elif 'K' in volume_str:
# #                             return float(volume_str.replace('K', '')) * 1000
# #                         else:
# #                             return float(volume_str)
# #                     df[col] = df[col].apply(convert_volume)
# #                 else:
# #                     df[col] = pd.to_numeric(df[col], errors='coerce')
        
# #         df = df.rename(columns={'Price': 'Close'})
# #         return df.dropna()  # Remove any rows with missing values
        
# #     except Exception as e:
# #         st.error(f"Error loading data: {str(e)}")
# #         return pd.DataFrame()

# # df = load_data()

# # # Check if data loaded successfully
# # if df.empty:
# #     st.error("No data available. Please check your data file.")
# #     st.stop()

# # # Sidebar controls
# # st.sidebar.header("Controls")
# # show_raw_data = st.sidebar.checkbox("Show raw data")

# # # Set default dates based on available data
# # min_date = df.index.min()
# # max_date = df.index.max()

# # start_date = st.sidebar.date_input(
# #     "Start date", 
# #     value=min_date,
# #     min_value=min_date,
# #     max_value=max_date
# # )
# # end_date = st.sidebar.date_input(
# #     "End date", 
# #     value=max_date,
# #     min_value=min_date,
# #     max_value=max_date
# # )

# # # Validate date range
# # if start_date > end_date:
# #     st.sidebar.error("End date must be after start date")
# #     st.stop()

# # # Filter data based on date selection
# # filtered_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

# # # If filtered data is empty, use the full dataset with warning
# # if filtered_df.empty:
# #     st.warning("No data available for selected date range. Showing full dataset instead.")
# #     filtered_df = df.copy()

# # # Tabs
# # tab1, tab2, tab3, tab4 = st.tabs(["Charts", "Correlation", "Model Performance", "Predictions"])

# # with tab1:
# #     # Candlestick chart
# #     st.subheader("Candlestick Chart with Volume")
    
# #     if not filtered_df.empty:
# #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
# #         df_reset = filtered_df.reset_index()
# #         df_reset['Date'] = df_reset['Date'].map(mdates.date2num)
        
# #         candlestick_ohlc(ax1, df_reset[['Date', 'Open', 'High', 'Low', 'Close']].values, 
# #                         width=0.6, colorup='g', colordown='r')
# #         ax1.xaxis_date()
# #         ax1.grid(True)
# #         ax1.set_title('Price Movement')
        
# #         ax2.bar(df_reset['Date'], df_reset['Vol.'], width=0.6, align='center')
# #         ax2.grid(True)
# #         ax2.set_ylabel('Volume')
        
# #         plt.tight_layout()
# #         st.pyplot(fig)
# #     else:
# #         st.warning("No data available for visualization")

# #     # Closing price line chart
# #     st.subheader("Closing Price Trend")
# #     if not filtered_df.empty and 'Close' in filtered_df.columns:
# #         fig, ax = plt.subplots(figsize=(12, 4))
# #         ax.plot(filtered_df.index, filtered_df['Close'], label='Closing Price', color='blue')
# #         ax.set_xlabel('Date')
# #         ax.set_ylabel('Price')
# #         ax.grid(True)
# #         st.pyplot(fig)
# #     else:
# #         st.warning("No closing price data available")

# # with tab2:
# #     # Correlation heatmap
# #     st.subheader("Feature Correlation Matrix")
# #     if not filtered_df.empty and len(filtered_df.columns) > 1:
# #         numeric_df = filtered_df.select_dtypes(include=[np.number])
# #         if len(numeric_df.columns) > 1:
# #             fig, ax = plt.subplots(figsize=(10, 8))
# #             corr_matrix = numeric_df.corr()
# #             sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
# #             st.pyplot(fig)
# #         else:
# #             st.warning("Not enough numeric columns for correlation analysis")
# #     else:
# #         st.warning("No data available for correlation analysis")

# # with tab3:
# #     # Model training and evaluation
# #     st.subheader("Model Training and Evaluation")
    
# #     if st.button("Train Models and Evaluate Performance"):
# #         if len(filtered_df) < 20:  # Minimum data points needed
# #             st.error(f"Not enough data points ({len(filtered_df)}). Please select a wider date range with at least 20 data points.")
# #         elif 'Close' not in filtered_df.columns:
# #             st.error("No closing price data available for modeling.")
# #         else:
# #             with st.spinner("Training models... This may take a few minutes"):
# #                 try:
# #                     # Prepare data for modeling
# #                     data = filtered_df['Close'].values.reshape(-1, 1)
                    
# #                     scaler = MinMaxScaler(feature_range=(0, 1))
# #                     scaled_data = scaler.fit_transform(data)
                    
# #                     # Create sequences
# #                     def create_sequences(data, seq_length):
# #                         X, y = [], []
# #                         for i in range(len(data)-seq_length-1):
# #                             X.append(data[i:(i+seq_length), 0])
# #                             y.append(data[i+seq_length, 0])
# #                         return np.array(X), np.array(y)
                    
# #                     seq_length = min(5, len(scaled_data) - 2)  # Ensure we have enough data
# #                     X, y = create_sequences(scaled_data, seq_length)
                    
# #                     # Split data
# #                     train_size = max(1, int(len(X) * 0.8))  # Ensure at least 1 sample
# #                     X_train, X_test = X[:train_size], X[train_size:]
# #                     y_train, y_test = y[:train_size], y[train_size:]
                    
# #                     # Reshape for LSTM
# #                     X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# #                     X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    
# #                     # Train LSTM model
# #                     lstm_model = Sequential()
# #                     lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
# #                     lstm_model.add(Dropout(0.2))
# #                     lstm_model.add(LSTM(50, return_sequences=True))
# #                     lstm_model.add(Dropout(0.2))
# #                     lstm_model.add(LSTM(50))
# #                     lstm_model.add(Dropout(0.2))
# #                     lstm_model.add(Dense(1))
                    
# #                     lstm_model.compile(optimizer='adam', loss='mean_squared_error')
# #                     lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
                    
# #                     # Make predictions
# #                     lstm_train_pred = lstm_model.predict(X_train_lstm)
# #                     lstm_test_pred = lstm_model.predict(X_test_lstm)
                    
# #                     # Inverse transform
# #                     lstm_train_pred = scaler.inverse_transform(lstm_train_pred)
# #                     y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
# #                     lstm_test_pred = scaler.inverse_transform(lstm_test_pred)
# #                     y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
# #                     # Calculate metrics
# #                     def calculate_metrics(y_true, y_pred):
# #                         mae = mean_absolute_error(y_true, y_pred)
# #                         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# #                         r2 = r2_score(y_true, y_pred)
# #                         return mae, rmse, r2
                    
# #                     lstm_train_mae, lstm_train_rmse, lstm_train_r2 = calculate_metrics(y_train_actual, lstm_train_pred)
# #                     lstm_test_mae, lstm_test_rmse, lstm_test_r2 = calculate_metrics(y_test_actual, lstm_test_pred)
                    
# #                     # Prepare data for traditional ML models
# #                     X_train_ml = X_train.reshape(X_train.shape[0], X_train.shape[1])
# #                     X_test_ml = X_test.reshape(X_test.shape[0], X_test.shape[1])
                    
# #                     # Define models
# #                     models = {
# #                         'Linear Regression': LinearRegression(),
# #                         'Decision Tree': DecisionTreeRegressor(random_state=42),
# #                         'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
# #                         'XGBoost': XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
# #                     }
                    
# #                     results = {}
                    
# #                     # Train and evaluate each model
# #                     for name, model in models.items():
# #                         model.fit(X_train_ml, y_train)
# #                         preds = model.predict(X_test_ml)
# #                         preds_scaled = scaler.inverse_transform(preds.reshape(-1, 1))
# #                         mae, rmse, r2 = calculate_metrics(y_test_actual, preds_scaled)
# #                         results[name] = {
# #                             'MAE': mae,
# #                             'RMSE': rmse,
# #                             'R2': r2,
# #                             'Predictions': preds_scaled
# #                         }
                    
# #                     # Add LSTM results
# #                     results['LSTM'] = {
# #                         'MAE': lstm_test_mae,
# #                         'RMSE': lstm_test_rmse,
# #                         'R2': lstm_test_r2,
# #                         'Predictions': lstm_test_pred
# #                     }
                    
# #                     # Display metrics comparison
# #                     st.subheader("Model Performance Comparison")
                    
# #                     # Metrics comparison chart
# #                     metrics = ['MAE', 'RMSE', 'R2']
# #                     model_names = list(results.keys())
                    
# #                     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    
# #                     for i, metric in enumerate(metrics):
# #                         values = [results[model][metric] for model in model_names]
# #                         if metric == 'R2':
# #                             bars = axes[i].bar(model_names, values, color='skyblue')
# #                             axes[i].set_title(f'{metric} (Higher is better)')
# #                         else:
# #                             bars = axes[i].bar(model_names, values, color='salmon')
# #                             axes[i].set_title(f'{metric} (Lower is better)')
                        
# #                         for bar in bars:
# #                             height = bar.get_height()
# #                             axes[i].text(bar.get_x() + bar.get_width()/2., height,
# #                                         f'{height:.2f}',
# #                                         ha='center', va='bottom')
                        
# #                         axes[i].set_ylabel(metric)
# #                         axes[i].tick_params(axis='x', rotation=45)
                    
# #                     plt.tight_layout()
# #                     st.pyplot(fig)
                    
# #                     # Metrics table
# #                     st.subheader("Detailed Metrics Table")
# #                     comparison_data = {
# #                         'Model': model_names,
# #                         'MAE': [results[m]['MAE'] for m in model_names],
# #                         'RMSE': [results[m]['RMSE'] for m in model_names],
# #                         'R2': [results[m]['R2'] for m in model_names]
# #                     }
                    
# #                     st.dataframe(pd.DataFrame(comparison_data).sort_values(by='RMSE'))
                    
# #                     # Actual vs Predicted plot
# #                     st.subheader("Actual vs Predicted Prices (Test Set)")
# #                     fig, ax = plt.subplots(figsize=(12, 6))
# #                     ax.plot(y_test_actual, label='Actual Prices', color='blue')
                    
# #                     # Plot predictions from each model
# #                     colors = ['green', 'orange', 'purple', 'brown', 'red']
# #                     for i, (name, result) in enumerate(results.items()):
# #                         ax.plot(result['Predictions'], label=f'{name} Predictions', 
# #                                color=colors[i], alpha=0.7, linestyle='--')
                    
# #                     ax.set_xlabel('Time Steps')
# #                     ax.set_ylabel('Price')
# #                     ax.legend()
# #                     ax.grid(True)
# #                     st.pyplot(fig)
                    
# #                 except Exception as e:
# #                     st.error(f"An error occurred during model training: {str(e)}")
# #                     st.stop()

# # with tab4:
# #     st.subheader("Future Price Prediction")
# #     st.info("""
# #     This feature would require implementing a forecasting function that predicts future prices based on the trained models.
# #     To implement this, we would need to:
# #     1. Save the trained models
# #     2. Create a forecasting function that generates future sequences
# #     3. Visualize the predictions with confidence intervals
# #     """)
# #     st.write("Feature coming soon...")

# # # Show raw data if selected
# # if show_raw_data and not filtered_df.empty:
# #     st.subheader("Raw Data")
# #     st.dataframe(filtered_df)
# # elif show_raw_data:
# #     st.warning("No data available to display")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mplfinance.original_flavor import candlestick_ohlc
# import matplotlib.dates as mdates

# # Page configuration
# st.set_page_config(page_title="Stock Price Analysis", layout="wide")

# # Title
# st.title(" Stock Price Prediction")

# # Load data
# @st.cache_data
# def load_data():
#     try:
#         df = pd.read_csv('Bharti Airtel Stock Price History.csv')
        
#         # Convert 'Date' to datetime and set as index
#         df['Date'] = pd.to_datetime(df['Date'])
#         df.set_index('Date', inplace=True)

#         # Convert string columns to numeric
#         numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
#         for col in numeric_cols:
#             if col in df.columns:
#                 # Remove commas and '%' signs
#                 df[col] = df[col].astype(str).str.replace(',', '', regex=True).str.replace('%', '', regex=True)

#                 if col == 'Vol.':
#                     def convert_volume(volume_str):
#                         if 'M' in volume_str:
#                             return float(volume_str.replace('M', '')) * 1000000
#                         elif 'K' in volume_str:
#                             return float(volume_str.replace('K', '')) * 1000
#                         else:
#                             return float(volume_str)
#                     df[col] = df[col].apply(convert_volume)
#                 else:
#                     df[col] = pd.to_numeric(df[col], errors='coerce')
        
#         df = df.rename(columns={'Price': 'Close'})
#         return df.dropna()  # Remove any rows with missing values
        
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return pd.DataFrame()

# df = load_data()

# # Check if data loaded successfully
# if df.empty:
#     st.error("No data available. Please check your data file.")
#     st.stop()

# # Sidebar controls
# st.sidebar.header("Controls")
# show_raw_data = st.sidebar.checkbox("Show raw data")

# # Set default dates based on available data
# min_date = df.index.min()
# max_date = df.index.max()

# start_date = st.sidebar.date_input(
#     "Start date", 
#     value=min_date,
#     min_value=min_date,
#     max_value=max_date
# )
# end_date = st.sidebar.date_input(
#     "End date", 
#     value=max_date,
#     min_value=min_date,
#     max_value=max_date
# )

# # Validate date range
# if start_date > end_date:
#     st.sidebar.error("End date must be after start date")
#     st.stop()

# # Filter data based on date selection
# filtered_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

# # If filtered data is empty, use the full dataset with warning
# if filtered_df.empty:
#     # st.warning("No data available for selected date range. Showing full dataset instead.")
#     filtered_df = df.copy()

# # Tabs
# tab1, tab2, tab3 = st.tabs(["Charts", "Correlation", "Metrics"])

# with tab1:
#     # Candlestick chart
#     st.subheader("Candlestick Chart with Volume")
    
#     if not filtered_df.empty:
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
#         df_reset = filtered_df.reset_index()
#         df_reset['Date'] = df_reset['Date'].map(mdates.date2num)
        
#         candlestick_ohlc(ax1, df_reset[['Date', 'Open', 'High', 'Low', 'Close']].values, 
#                         width=0.6, colorup='g', colordown='r')
#         ax1.xaxis_date()
#         ax1.grid(True)
#         ax1.set_title('Price Movement')
        
#         ax2.bar(df_reset['Date'], df_reset['Vol.'], width=0.6, align='center')
#         ax2.grid(True)
#         ax2.set_ylabel('Volume')
        
#         plt.tight_layout()
#         st.pyplot(fig)
#     else:
#         st.warning("No data available for visualization")

#     # Closing price line chart
#     st.subheader("Closing Price Trend")
#     if not filtered_df.empty and 'Close' in filtered_df.columns:
#         fig, ax = plt.subplots(figsize=(12, 4))
#         ax.plot(filtered_df.index, filtered_df['Close'], label='Closing Price', color='blue')
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Price')
#         ax.grid(True)
#         st.pyplot(fig)
#     else:
#         st.warning("No closing price data available")

# with tab2:
#     # Correlation heatmap
#     st.subheader("Feature Correlation Matrix")
#     if not filtered_df.empty and len(filtered_df.columns) > 1:
#         numeric_df = filtered_df.select_dtypes(include=[np.number])
#         if len(numeric_df.columns) > 1:
#             fig, ax = plt.subplots(figsize=(10, 8))
#             corr_matrix = numeric_df.corr()
#             sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
#             st.pyplot(fig)
#         else:
#             st.warning("Not enough numeric columns for correlation analysis")
#     else:
#         st.warning("No data available for correlation analysis")

# with tab3:
#     st.subheader("Performance Metrics")
    
#     # Sample metrics (replace with your actual metrics)
#     metrics_data = {
#         'Model': ['LSTM', 'Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
#         'MAE': [12.34, 15.67, 14.89, 13.45, 12.78],
#         'RMSE': [18.76, 22.34, 20.15, 19.23, 18.89],
#         'R2': [0.92, 0.85, 0.88, 0.90, 0.91]
#     }
    
#     metrics_df = pd.DataFrame(metrics_data)
    
#     # Display metrics table
#     st.dataframe(metrics_df.sort_values(by='RMSE'))
    
#     # Metrics comparison chart
#     st.subheader("Metrics Comparison")
    
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
#     for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
#         values = metrics_df[metric]
#         if metric == 'R2':
#             bars = axes[i].bar(metrics_df['Model'], values, color='skyblue')
#             axes[i].set_title(f'{metric} (Higher is better)')
#         else:
#             bars = axes[i].bar(metrics_df['Model'], values, color='salmon')
#             axes[i].set_title(f'{metric} (Lower is better)')
        
#         for bar in bars:
#             height = bar.get_height()
#             axes[i].text(bar.get_x() + bar.get_width()/2., height,
#                         f'{height:.2f}',
#                         ha='center', va='bottom')
        
#         axes[i].set_ylabel(metric)
#         axes[i].tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
#     st.pyplot(fig)

# # Show raw data if selected
# if show_raw_data and not filtered_df.empty:
#     st.subheader("Raw Data")
#     st.dataframe(filtered_df)
# elif show_raw_data:
#     st.warning("No data available to display")




# # # stock_app.py
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import plotly.graph_objects as go

# # # Load and prepare data
# # @st.cache_data
# # def load_data():
# #     def convert_volume(vol_str):
# #         if isinstance(vol_str, str):
# #             vol_str = vol_str.replace(',', '')
# #             if 'M' in vol_str:
# #                 return float(vol_str.replace('M', '')) * 1_000_000
# #             elif 'K' in vol_str:
# #                 return float(vol_str.replace('K', '')) * 1_000
# #         return float(vol_str)

# #     df = pd.read_csv('Reliance Industries Stock Price History.csv',
# #                      parse_dates=['Date'],
# #                      index_col='Date')
    
# #     numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    
# #     for col in numeric_cols:
# #         if col == 'Vol.':
# #             df[col] = df[col].apply(convert_volume)
# #         else:
# #             df[col] = df[col].astype(str).str.replace(',', '').str.replace('%', '').astype(float)
    
# #     return df.sort_index()

# # df = load_data()

# # # Rest of your app code remains the same...
# # # [Keep the visualization and other sections unchanged]

# # # App header
# # st.title('📈 Reliance Industries Stock Analysis')
# # st.write("Full historical data analysis")

# # # Main display
# # st.success(f"Showing complete dataset: {df.index.min().date()} to {df.index.max().date()}")

# # # Candlestick chart
# # st.subheader("Candlestick Chart with Volume")
# # fig = go.Figure(data=[
# #     go.Candlestick(
# #         x=df.index,
# #         open=df['Open'],
# #         high=df['High'],
# #         low=df['Low'],
# #         close=df['Price'],
# #         name='Price'
# #     ),
# #     go.Bar(
# #         x=df.index,
# #         y=df['Vol.'],
# #         name='Volume',
# #         yaxis='y2'
# #     )
# # ])

# # fig.update_layout(
# #     title='Price Movement',
# #     yaxis_title='Price (₹)',
# #     xaxis_rangeslider_visible=False,
# #     yaxis2=dict(
# #         title='Volume',
# #         overlaying='y',
# #         side='right'
# #     ),
# #     height=600
# # )

# # st.plotly_chart(fig, use_container_width=True)

# # # Correlation matrix
# # st.subheader("Correlation Metrics")
# # corr_matrix = df[['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']].corr()
# # st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None))

# # # Price prediction section
# # st.subheader("Price Prediction")
# # # Add your LSTM/XGBoost prediction code here

# # # Data summary
# # st.subheader("Data Summary")
# # st.write(df.describe())
