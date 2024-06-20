import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data until June 2023
data = pd.DataFrame({
    'Product Name': ['Engine Oil', 'Brake Pads', 'Air Filter', 'Spark Plugs', 'Battery'] * 60,
    'Category': ['Maintenance', 'Brakes', 'Filters', 'Ignition', 'Electrical'] * 60,
    'Subcategory': ['Fluids', 'Brake System', 'Air Intake', 'Ignition Coils', 'Batteries'] * 60,
    'Date': pd.date_range('2018-01-01', periods=300, freq='M'),
    'Amount to be Paid': np.random.randint(low=1000, high=10000, size=300)
})

# Ensure that every product has a demand in each month until June 2023
product_months = pd.MultiIndex.from_product([data['Product Name'].unique(), pd.date_range('2018-01-01', '2023-06-01', freq='M')], names=['Product Name', 'Date'])
data = data.set_index(['Product Name', 'Date']).reindex(product_months).reset_index()
data['Units Purchased'] = np.random.randint(low=0, high=100, size=len(data))

# Aggregate the data to get the total sales per month
monthly_sales = data.groupby('Date')['Units Purchased'].sum().reset_index()

# Set the Date column as the index
monthly_sales = monthly_sales.set_index('Date')

# Function to implement Croston's method for forecasting
def croston_forecast(y, alpha, beta, periods):
    n = len(y)
    a = np.zeros(n)
    b = np.zeros(n)
    croston_forecast = np.zeros(n + periods)

    demand = np.array(y)

    for i in range(1, n):
        if demand[i] > 0:
            a[i] = alpha * demand[i] + (1 - alpha) * a[i - 1]
            b[i] = beta * (1 - alpha) * demand[i] / a[i] + (1 - beta) * b[i - 1]
        else:
            a[i] = a[i - 1]
            b[i] = b[i - 1]

    croston_forecast[:n] = a + b
    croston_forecast[n:] = a[-1] + b[-1] * np.arange(1, periods + 1)

    return croston_forecast

# Set the parameters for Croston's method
alpha = 0.1
beta = 0.1
forecast_periods = 18

# Perform the sales forecast using Croston's method
croston_forecast = croston_forecast(monthly_sales['Units Purchased'], alpha, beta, forecast_periods)

# Create the forecasted date range
forecast_dates = pd.date_range(start='2023-07-01', periods=forecast_periods, freq='M')

# Printing the Forecasted Sales

print('Forecasted Sales:')
for date, sales in zip(forecast_dates, croston_forecast[-forecast_periods:]):
    print(f'{date}: {sales:.2f}  ')

# Plot the sales forecast
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales['Units Purchased'], label='Historical Sales')
plt.plot(forecast_dates, croston_forecast[-forecast_periods:], label='Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Units Purchased')
plt.title('Sales Forecast (Croston\'s Method)')
plt.legend()
plt.show()

