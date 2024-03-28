import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
#Import the given data
df = pd.read_csv('sample_10.csv')
#Make necessary modifications
df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])
df = df.drop(columns=['Unnamed: 0'])
#Check for any missing values
mis = df.isna().sum()
print(mis)
#If yes, then the missing values is been imputed by mean
df.fillna(df.mean(),inplace=True)
mis = df.isna().sum()
print(mis)
def checkpoint1():
    #Visualizing the data for better understanding and for further analysis
    plt.plot(df['point_timestamp'],df['point_value'])
    plt.xticks(rotation=45)
    plt.figure(figsize=(8, 4))
    plt.show
    
    #Performing the Augmented Dickey-Fuller test for checking the stationarity of the given dataset

    result = adfuller(df['point_value'])

    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    l=[]
    for key, value in result[4].items():
        l.append(value)
        
        print(f'   {key}: {value}')
    # Check stationarity
    d=0
    if result[0] < l[0] and result[1] < 0.05:
        print("The data is stationary at the 1% significance level.")
    elif result[0] < l[1] and result[1] < 0.05:
        print("The data is stationary at the 5% significance level.")
    elif result[0] < l[2] and result[1] < 0.05:
        print("The data is stationary at the 10% significance level.")
    else:
        print("The data is not stationary.")
        d=1
    

    def test_seasonality(data, max_lag):
        

        # Plot autocorrelation function (ACF)
        plot_acf(data, lags=max_lag)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Function (ACF)')

        # Check for significant spikes at seasonal lags
        for lag in range(1, max_lag + 1):
            if abs(data.autocorr(lag)) >= 0.2:  # Adjust significance threshold as needed
                plt.axvline(x=lag, color='red', linestyle='--')
                plt.text(lag, 0.2, f'Lag {lag}', rotation=90, color='red')

        # Display plot
        plt.show()

        # Check if seasonality is detected
        seasonality_detected = any(abs(data.autocorr(lag)) >= 0.2 for lag in range(1, max_lag + 1))
        
        return seasonality_detected

    # Example usage:
    # Assuming 'data' is your time series data (Pandas Series or DataFrame)
    seasonality_detected = test_seasonality(df['point_value'], max_lag=30)  # Test for seasonality with maximum lag of 12
    print("Seasonality Detected:", seasonality_detected)

    

    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.preprocessing import MinMaxScaler

    def create_ma_model_and_calculate_mape(data, train_size=0.7, forecast_steps=7):
       

        # Step 2: Convert data to pandas Series if it's not already
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        # Step 3: Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(data.values.reshape(-1, 1))

        # Step 4: Split the normalized data into training and test sets
        train_size = int(len(normalized_data) * train_size)
        train_data, test_data = normalized_data[:train_size], normalized_data[train_size:]

        # Step 5: Fit a Moving Average model on the training data
        model = SARIMAX(train_data, order=(0, 0, 3), seasonal_order=(0, 0, 0, 12))  # SARIMA(1,1,1)(0,0,0,12)
        fitted_model = model.fit()

        # Step 6: Predict values on the test data using the trained model
        predictions = fitted_model.predict(start=len(train_data), end=len(normalized_data)-1, typ='levels')

        # Step 7: Inverse transform the predictions to the original scale
        predictions_original_scale = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        test_data_original_scale = scaler.inverse_transform(test_data.reshape(-1, 1)).flatten()

        # Step 8: Calculate the Mean Absolute Percentage Error (MAPE)
        mape = mean_absolute_percentage_error(test_data_original_scale, predictions_original_scale)
        aic = fitted_model.aic

        # Calculate BIC
        bic = fitted_model.bic
        print(aic)
        print(bic)
        # Step 9: Forecast future values
        future_forecast = fitted_model.forecast(steps=forecast_steps)
        # Step 10: Inverse transform the forecasted values to the original scale
        future_forecast_original_scale = scaler.inverse_transform(future_forecast.reshape(-1, 1)).flatten()

        return mape, future_forecast_original_scale

    
    mape, next_7_days_forecast = create_ma_model_and_calculate_mape(df['point_value'], forecast_steps=7)
    print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))
    print("Forecast for the next day:", next_7_days_forecast)





    if mape > 0.3:
        score = min(score, 0)

    score = min(max(0, score), 10)

    return score
from prophet import Prophet
import pandas as pd

# Load and preprocess time series data
# Assuming df contains a datetime column "ds" and a numerical column "y"

# Split data into overlapping windows of 7 days each
windows = [df[i:i+7] for i in range(len(df)-6)]

# Initialize Prophet model
model = Prophet()

# Iterate over each window
for window_df in windows:
    # Fit the model to the window of data
    model.fit(window_df)
    
    # Make predictions for the window
    future = model.make_future_dataframe(periods=len(window_df))  # Forecasting for the same length as the window
    forecast = model.predict(future)
    
    # Calculate residuals
    residuals = window_df['y'] - forecast['yhat']
    
    # Define threshold for anomaly detection (e.g., 3 standard deviations from the mean)
    threshold = residuals.mean() + 3 * residuals.std()
    
    # Detect anomalies
    anomalies = window_df[abs(residuals) > threshold]
    
    # Print or store anomalies for each window
    print("Anomalies in window:", anomalies)

# Aggregate and visualize results as needed
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict_data(data: List[float]):

    # preprocess the data
    data = np.array(data)

    # determine the optimal threshold for forecastability
    mape,score = create_sarima_model_and_calculate_mape(data, train_size=0.7)

   

    # generate predictions
    predictions = model.predict(data)

    # calculate the MAPE
    mape = np.mean(np.abs((data - predictions) / data))

    # return the predictions, forecastability score, and MAPE
    return {'score': score, 'mape': mape}
import time

# define the timeout for the sweeping algorithm
TIMEOUT = 10

# define the range of parameter tuning
WINDOW_SIZES = range(3, 20)
BUFFER_SIZES = range(100, 1000, 100)
BATCH_SIZES = range(10, 100, 10)

def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time

    return result, elapsed_time

best_settings = None
best_time = TIMEOUT
for window_size in WINDOW_SIZES:
    for buffer_size in BUFFER_SIZES:
        for batch_size in BATCH_SIZES:
            model, time_taken = time_function(train_model, data, window_size=window_size, buffer_size=buffer_size, batch_size=batch_size)

            if time_taken < best_time:
                best_time = time_taken
                best_settings = {"window_size": window_size, "buffer_size": buffer_size, "batch_size": batch_size}


def detect_changepoint(data, window_size):
    # Calculate the Augmented Dickey-Fuller test statistic
    test_stat, _, _, _, _, _ = adfuller(data, maxlag=window_size, autolag=False, regression='c')

    # Determine the changepoint by comparing the ADF statistic with the critical value
    if test_stat < -3.43:
        return window_size
    return window_size // 2

def batch_reduce_detection(data):
    # Initialize the minimum window size
    min_window_size = 7

    # Determine the optimal window size
    for _ in range(10):
        window_size = detect_changepoint(data, min_window_size)
        if window_size < min_window_size:
            break

        min_window_size = window_size

    return min_window_size