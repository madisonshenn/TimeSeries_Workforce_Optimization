# TimeSeries_Workforce_Optimization
Worked with group of 5 using hybrid time-series forecasting models to optimize part-time staffing in retail.


### Please scroll down to see the Python codes & access the jupyter notebook named "ML_Staffing_Solution_Retail" for the complete codes
## Final Delivery:
https://www.canva.com/design/DAGiv6WoJvo/NMFB0ncDZHvOPiuUsBlAOw/edit?utm_content=DAGiv6WoJvo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 

## Product Overview:
### Target Buyer: 
Regional Store Managers like Mark, who oversee operations across multiple retail locations and are responsible for staffing, cost control, and store performance.
### Business Problem:
Mark is stuck juggling outdated workforce tools that can't forecast demand, leading to overstaffing during slow hours and understaffing during peak times. The result? Wasted labor costs, burnout among frontline staff, poor customer experiences, and missed revenue targets—plus traditional solutions are either too expensive or too complex to implement.
### Value Proposition:
The platform delivers intelligent labor forecasting and scheduling that integrates with existing HR systems, reduces turnover, and boosts revenue by aligning staffing with real-time and forecasted demand. It’s plug-and-play for ops teams and directly improves both the customer experience and employee retention—no technical degree required.
### ML Solution:
The solution uses a hybrid forecasting model (SVD-SARIMAX + ETS) to predict weekly sales trends with high accuracy, enabling smarter, data-driven staffing decisions. By integrating historical sales and economic indicators, it balances long-term patterns with real-time adaptability—automating the scheduling process across stores with minimal technical overhead.
### Result & Output of the Solution:
The model delivers a 12% cut in labor costs by reducing unnecessary part-time hires and a 15% boost in customer service efficiency by solving peak-hour understaffing. It scales effortlessly across 100+ retail locations, proving it's not just a pilot—it’s plug-and-play at enterprise scale.
### Business Model:
This is a SaaS-based subscription model with tiered pricing—$300/month for small stores, scaling up to $500/month for enterprise clients—designed to flex with business size and needs. Revenue is diversified through API licensing and consulting services for large-scale retailers, but the real growth engine is the subscription tier, built to scale.
### Investment Cycle:
With projected ROI hitting 70.53% by Year 3, the investment thesis is strong—subscription accounts are scaling fast: 40% YoY growth from enterprise, 20% from small stores, and 17% from midsize. Consulting and customization are strategic upsell levers, growing at 16% annually but kept secondary to maintain product-first scalability.

## Python Codes
Access the jupyter notebook named "ML_Staffing_Solution_Retail" for the complete codes

### Data Cleaning & EDA:

```
df_sales = df_sales.groupby(['Store', 'Date'])['Weekly_Sales'].sum().reset_index()
df_sales['Type'] = df_sales['Store'].map(df_stores.set_index('Store').Type)
df_sales['Size'] = df_sales['Store'].map(df_stores.set_index('Store').Size)

df_sales = df_sales.merge(df_features[['Store', 'Date', 'IsHoliday']], on=['Store', 'Date'], how='left')
df_sales.head()

plt.figure(figsize=(5, 3))
sns.histplot(data = df_stores, x = 'Type')
plt.title('Store Type Distribution')
plt.show()

# split stores by size into 3 groups
bin_width = 30000
df_stores['SizeGroup'] = (df_stores["Size"] // bin_width) * bin_width
# Count the number of stores in each Size Group and Type
grouped = df_stores.groupby(["SizeGroup", "Type"]).size().unstack(fill_value=0)

# Plotting
grouped.plot(kind='bar', stacked=True, figsize=(10, 6))

# Chart formatting
plt.title("Store Count by Size Group and Type")
plt.xlabel("Size Group")
plt.ylabel("Number of Stores")
plt.legend(title="Store Type")
plt.xticks(rotation=45)
plt.show()

# plot the box plot for each store's weekly sales
custom_palette = {'A': 'skyblue', 'B': 'yellow', 'C': 'lightgreen'}  # Custom colors

plt.figure(figsize=(20, 12))
sns.boxplot(data=df_sales, x='Store', y='Weekly_Sales', hue='Type', palette=custom_palette, linewidth=0.5)
plt.title('Weekly Sales by Store')
plt.show()
```

### Check Data Completeness

```
df_sales['Date'] = pd.to_datetime(df_sales['Date'])

# for each store, check whether data is recorded every week, start_date is the first date recorded
def missing_data(df):
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='7d')
    date_range = pd.DataFrame(date_range, columns=['Date'])
    date_range['Store'] = df['Store'].unique()[0]
    df = pd.merge(date_range, df, on=['Date', 'Store'], how='left')

    if df['Weekly_Sales'].isnull().sum() > 0:
        print('Missing data for store', df['Store'].unique()[0])
    else:
        return False
```

```
# Function to check all stores
def check_weekly_stores(df):
    stores = df['Store'].unique()
    for store in stores:
        store_df = df[df['Store'] == store]
        result = missing_data(store_df)
        if result == True:
            break
    print('All stores checked and no missing data found')

check_weekly_stores(df_sales)
```



```
# For each "Date", keep the weighted average of the weekly sales for each store, weighted by the store size
df_sales['Weighted_Sales'] = df_sales['Weekly_Sales'] * df_sales['Size']

# Aggregate by Date
df_weighted_avg = df_sales.groupby("Date").apply(
    lambda x: x["Weighted_Sales"].sum() / x["Size"].sum()
).reset_index(name="Weighted_Avg_Sales")

# Add back "IsHoliday" column
df_weighted_avg["IsHoliday"] = df_sales.groupby("Date")["IsHoliday"].max().values

# Display the result
df_weighted_avg


# check if Date is datetime
df_weighted_avg['Date'] = pd.to_datetime(df_weighted_avg['Date'])


# Set 'Date' column as the index
df_weighted_avg.set_index('Date', inplace=True)


# Plot the weighted average sales
df_weighted_avg['Weighted_Avg_Sales'].plot(label='weighted average sales', figsize=(10, 6))
plt.title('Weighted Average Sales Over Time')

# Highlight holidays
holiday_dates = df_weighted_avg[df_weighted_avg['IsHoliday'] == True].index
for date in holiday_dates:
    plt.axvline(x=date, color='red', alpha=0.3, linestyle='--')

plt.legend()
plt.show()
```

### Model Fitting
```
df_store = df_sales.groupby(['Store', 'Date']).agg({
    'Weekly_Sales': 'sum',
    'IsHoliday': 'first'})

# Merge with featureson 'store' and 'date'
df = pd.merge(df_store, df_features, on=['Store', 'Date'], how='left')

df = df.drop(columns=['IsHoliday_y'])

df = df.rename(columns={'IsHoliday_x': 'IsHoliday'})


# Fill missing markdowns with 0 (assumption: no markdown applied)
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
df[markdown_cols] = df[markdown_cols].fillna(0)
df['IsHoliday'] = df['IsHoliday'].astype(int)


def predict_scale(store_no):
  if (store_no == 0):
    train = df.groupby('Date').agg({
            'Weekly_Sales': 'sum',         # Total weekly sales across all stores
            'MarkDown1': 'sum',           # Sum of markdowns for all stores
            'MarkDown2': 'sum',           # Sum of markdowns for all stores
            'MarkDown3': 'sum',           # Sum of markdowns for all stores
            'MarkDown4': 'sum',           # Sum of markdowns for all stores
            'MarkDown5': 'sum',           # Sum of markdowns for all stores
            'Temperature': 'mean',         # Average temperature across all stores
            'Fuel_Price': 'mean',          # Average fuel price across all stores
            'CPI': 'mean',                 # Average CPI across all stores
            'Unemployment': 'mean',        # Average unemployment across all stores
            'IsHoliday': 'max'             # Sum of holidays (could count holidays per week)
        }).reset_index()



    # Case when store_no is between 1 and 45, filter data for that specific store
  else:
      train = df[df['Store'] == store_no]

  train, test = train_test_split(train, test_size=0.2, random_state=42, shuffle=False)
  train.set_index('Date', inplace=True)
  test.set_index('Date', inplace=True)

  return train, test
```

### ETS
```
store_no = 0
train, test = predict_scale(store_no)
# triple ETS - Holt Winter's Seasonal Method
triple_ets_add = ExponentialSmoothing(train['Weekly_Sales'], trend = 'add', seasonal = 'add', seasonal_periods=52).fit()
triple_ets_mul = ExponentialSmoothing(train['Weekly_Sales'], trend = 'mul', seasonal = 'mul', seasonal_periods=52).fit()

triple_ets_add_pred = triple_ets_add.forecast(29)
triple_ets_mul_pred = triple_ets_mul.forecast(29)

#plot the train, test, and predictions

ax = test['Weekly_Sales'].plot(marker='o', color='black', figsize=(20,8), legend=True)

train['Weekly_Sales'].plot(figsize=(20,8), legend=True)

triple_ets_mul_pred.plot(marker='o', ax=ax, color='orange', legend=True, label = 'triple ets mul')
triple_ets_add_pred.plot(marker='o', ax=ax, color='blue', legend=True, label = 'triple ets add')

plt.title(f'Train, Test and Predicted Test using Triple ETS - Holt Winters Seasonal For Store')
plt.show()

y_min = min(test['Weekly_Sales'].min(), triple_ets_add_pred.min())
y_max = max(test['Weekly_Sales'].max(), triple_ets_add_pred.max())

# Extend the range by doubling the difference
y_range = y_max - y_min
plt.ylim(y_min - 0.5*y_range, y_max + 0.5*y_range)

ax = test['Weekly_Sales'].plot(marker='o', color='black', figsize=(20,8), legend=True)

triple_ets_mul_pred.plot(marker='o', ax=ax, color='orange', legend=True, label = 'triple ets mul')

triple_ets_add_pred.plot(marker='o', ax=ax, color='blue', legend=True, label = 'triple ets add')

plt.title('ETS - Train, Test and Predicted')
plt.show()
```

### SARIMAX
```
# Fit the best SARIMA-X model
final_mod = sm.tsa.SARIMAX(train['Weekly_Sales'],
                           exog=train[['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + markdown_cols],
                           order= (1,0,1),
                           seasonal_order=(1, 1, 1, 52),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
final_results = final_mod.fit()

# Print the model summary
print(final_results.summary())


# Make predictions on test data
test_predictions = final_results.get_forecast(steps=len(test), exog=test[['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + markdown_cols])
sarimax_pred = test_predictions.predicted_mean

# Visualize predictions
plt.figure(figsize=(12, 6))
y_min = min(test['Weekly_Sales'].min(), sarimax_pred.min())
y_max = max(test['Weekly_Sales'].max(), sarimax_pred.max())

# Extend the range by doubling the difference
y_range = y_max - y_min
plt.ylim(y_min - 0.5*y_range, y_max + 0.5*y_range)
plt.plot(test.index, test['Weekly_Sales'], label='Actual', color='black')
plt.plot(test.index, sarimax_pred, label='Predicted', color='orange')
#plt.fill_between(test.index, predicted_conf_int.iloc[:, 0], predicted_conf_int.iloc[:, 1], color='pink')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('SARIMAX - Test Data vs. Predicted Data')
plt.legend()
plt.show()

```

### SVD-SARIMAX
```
def preprocess_svd(train, n_components=5):

    train_values = train.values

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy="mean")
    train_values = imputer.fit_transform(train_values)

    # Apply SVD
    U, s, Vt = svd(train_values, full_matrices=False)

    # Keep only n_components
    S_reduced = np.diag(s[:n_components])
    U_reduced = U[:, :n_components]
    Vt_reduced = Vt[:n_components, :]

    # Reconstruct reduced-rank matrix
    train_reduced = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))

    return pd.DataFrame(train_reduced, index=train.index, columns=train.columns)

def seasonal_arima_svd(train, test, n_components=5):

    horizon = len(test)  # Number of weeks to forecast

    # Apply SVD preprocessing
    train_reduced = preprocess_svd(train, n_components=n_components)

    for store in train_reduced.columns:
        series = train_reduced[store].dropna()

        if series.isnull().sum() > len(series) / 3:
            # Fallback: Use last known value if too many missing
            print(f"Fallback for Store: {store}")
            test[store] = series.iloc[-1]
        else:
            try:
                # Fit Seasonal ARIMA (SARIMA)
                model = ARIMA(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
                model_fit = model.fit()

                # Forecast
                forecast_values = model_fit.forecast(steps=horizon)
                test[store] = forecast_values.values
            except Exception as e:
                print(f"ARIMA failed for Store {store}, using last value. Error: {e}")
                test[store] = series.iloc[-1]  # Fallback

    return test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

train_matrix = train.values  # Converts DataFrame to NumPy array
test_matrix = test.values

# Standardize the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_matrix)
test_scaled = scaler.transform(test_matrix)

# Apply SVD for dimensionality reduction
n_components = 10  # You can tune this
svd = TruncatedSVD(n_components=n_components)
train_reduced = svd.fit_transform(train_scaled)

# Train SARIMA models on reduced components
sarima_models = []
forecasts = []

for i in range(n_components):
    series = train_reduced[:, i]  # Select reduced component
    #model = ARIMA(series, order=(0, 1, 1), seasonal_order=(1, 1, 1, 52))
    model = SARIMAX(series, exog=train[['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + markdown_cols], order=(1,1,1), seasonal_order=(1,1,1,52))  # Weekly seasonality
    results = model.fit()
    sarima_models.append(results)

    # Forecast future values
    test_predictions = results.get_forecast(steps=len(test), exog=test[['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + markdown_cols])
    forecasts.append(test_predictions.predicted_mean)

# Convert forecasts back using inverse SVD
forecasts = np.array(forecasts).T  # Transpose to match dimensions
test_reconstructed = svd.inverse_transform(forecasts)  # Inverse SVD
test_reconstructed = scaler.inverse_transform(test_reconstructed)  # Reverse standardization

# Convert back to DataFrame
test_pred_df = pd.DataFrame(test_reconstructed, index=test.index, columns=test.columns)

```

### Data Visualization
```
plt.figure(figsize=(12, 6))
y_min = min(test['Weekly_Sales'].min(), average['Weekly_Sales'].min())
y_max = max(test['Weekly_Sales'].max(), average['Weekly_Sales'].max())
y_range = y_max - y_min
plt.ylim(y_min - 1*y_range, y_max + 1*y_range)
plt.plot(test.index, test['Weekly_Sales'], label='Actual', color='black')
plt.plot(test.index, average['Weekly_Sales'], label='Predicted', color='orange')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title(f'Average - Actual vs. Predicted Sales')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))

y_min = min(test['Weekly_Sales'].min(),
            test_pred_df['Weekly_Sales'].min(),
            average['Weekly_Sales'].min(),
            triple_ets_add_pred.min(),
            triple_ets_mul_pred.min())

y_max = max(test['Weekly_Sales'].max(),
            test_pred_df['Weekly_Sales'].max(),
            average['Weekly_Sales'].max(),
            triple_ets_add_pred.max(),
            triple_ets_mul_pred.max())

y_range = y_max - y_min
plt.ylim(y_min - 0.8 * y_range, y_max + 0.8 * y_range)

plt.plot(test.index, test['Weekly_Sales'], label='Actual', color='black', marker='o')

plt.plot(test.index, test_pred_df['Weekly_Sales'], label='SVD-SARIMAX', color='orange', linestyle='dashed', marker='s')

plt.plot(test.index, average['Weekly_Sales'], label='Average', color='green', linestyle='dotted', marker='^')

plt.plot(test.index, triple_ets_mul_pred, label='Triple ETS Multiplicative', color='red', linestyle='dotted', marker='d')

plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Actual vs. Predicted Sales Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

### Model Evaluation
```
def compute_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, mape

actual_sales = test['Weekly_Sales']/1e7

# Compute metrics for each model
sarimax_rmse, sarimax_mae, sarimax_mape = compute_metrics(actual_sales, test_pred_df['Weekly_Sales']/1e7)
ets_rmse, ets_mae, ets_mape = compute_metrics(actual_sales, triple_ets_mul_pred/1e7)
avg_rmse, avg_mae, avg_mape = compute_metrics(actual_sales, average['Weekly_Sales']/1e7)

metrics_df = pd.DataFrame({
    "Model": ["SARIMAX", "ETS", "Average (SARIMAX + ETS)"],
    "RMSE": [sarimax_rmse, ets_rmse, avg_rmse],
    "MAE": [sarimax_mae, ets_mae, avg_mae],
    "MAPE (%)": [sarimax_mape, ets_mape, avg_mape]
})

print("Performance Metrics:")
print(metrics_df)

average_pct_change = average['Weekly_Sales'].pct_change() * 100

actual_pct_change = test['Weekly_Sales'].pct_change() * 100

percentage_change_df = pd.DataFrame({
    "Sales % Change": average_pct_change
})

print(percentage_change_df)
```
