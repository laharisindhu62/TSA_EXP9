# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 11/05/25

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```

# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = '/mnt/data/powerconsumption.csv'
data = pd.read_csv(file_path)

# Convert 'Datetime' column to datetime format
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m/%d/%Y %H:%M')

# Set 'Datetime' column as index
data.set_index('Datetime', inplace=True)

# Define the ARIMA model function
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit the ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Forecast
    forecast = fitted_model.forecast(steps=len(test_data))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data', color='blue')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data', color='green')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='orange')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    # Print RMSE
    print("Root Mean Squared Error (RMSE):", rmse)

# Fit the ARIMA model for Power Consumption Zone 1
arima_model(data, 'PowerConsumption_Zone1', order=(5, 1, 0))
```
### OUTPUT:

![image](https://github.com/user-attachments/assets/a4f36c89-bea8-40da-9b9c-ab6ca8c9b95a)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
