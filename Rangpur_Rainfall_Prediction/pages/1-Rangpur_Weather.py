import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np

# Get current directory
current_dir = os.path.dirname(__file__)

# Construct path to artifactory directory
artifactory_dir = os.path.join(current_dir, '..', 'artifactory')

# Construct path for the files
model_path = os.path.join(artifactory_dir, 'xgboost_rangpur_model.joblib')
testdata_path = os.path.join(artifactory_dir, 'rangpur_test.csv')
image_path = os.path.join(artifactory_dir, 'Rangpur_1.jpg')

# Load the model and test data
model = joblib.load(model_path )
test_data = pd.read_csv(testdata_path, index_col='Date', parse_dates=True)

# Generate predictions
y_pred = np.abs(model.predict(test_data.drop(columns=['PRECTOTCORR'])))


st.title('Rainfall Prediction in Rangpur')

# Placeholder for image of Rangpur
st.image(image_path, caption='Tajhat Palace in Rangpur')  

st.header('Motivation & Goal:')
st.write('Rangpur is a major city in northwestern Bangladesh located along the Ghaghat River and is part of the Rangpur Division. Since it is known to be affected by flooding, the average precipitation for the city is provided based on the weather pattern.')

st.header('Model & Result:')
st.write('The XGBoost model after hyper-parameter tuning shows promising results. Some key metrics of the model on test data are as follows:')
st.write('1. RMSE: 5.2992 mm per day')
st.write('2. MAE: 2.8471 mm per day (A better suited metric)')
st.write('3. R^2: 72.98%') 

st.header('Key Finding:')
st.write('The amount of rainfall from the previous day (i.e., the newly devised feature called precip_mean_1) is the strongest predictor of the next day’s rainfall.')

st.header('Conclusion:')
st.write('Given the distribution of precipitation with a daily average precipitation of 6.66 mm and a standard deviation of 12.80 mm, these metrics indicate a model with reasonable predictive power.')

st.header('Future Work Suggestion:')
st.write("1. Since there are extreme precipitation events on certain days, incorporate additional data sources to improve model robustness.")
st.write("2. Further improve the feature engineering process to capture complex rainfall patterns.")	

# Display the table with predicted values
st.subheader('Forecast Table')

# Sort the DataFrame by the index (which is the Date) in descending order
forecast_data = test_data.copy()
forecast_data['Predicted Precipitation (mm)'] = y_pred
forecast_data.sort_index(ascending=False, inplace=True)

# Use the slider to determine the number of forecast days to display
forecast_days = st.slider('Number of Forecast Days', 1, len(forecast_data), 30)

# Select the top 'forecast_days' entries after sorting
forecast_display = forecast_data.head(forecast_days)
forecast_display['Date'] = forecast_display.index.strftime('%Y-%m-%d')  # Format the date without the time component

# Display the DataFrame in the Streamlit app
st.write(f"Forecasting for the most recent {forecast_days} days")
st.write(forecast_display[['Date', 'Predicted Precipitation (mm)']].reset_index(drop=True))

# Plot comparing predicted and actual precipitation
st.subheader('Precipitation Forecast vs. Actual Plot')
fig, ax = plt.subplots()
# Formatter for the date on the x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)  # Rotate the dates for better visibility
ax.plot(forecast_data.index, forecast_data['PRECTOTCORR'], label='Actual', marker='o', color='blue')
ax.plot(forecast_data.index, forecast_data['Predicted Precipitation (mm)'], label='Predicted', marker='x', color='red', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Precipitation (mm)')
ax.legend()
plt.tight_layout()  # Improve layout to accommodate rotated x-axis labels
st.pyplot(fig)


st.header("Monsoon Season Precipitation")
# Given the importance of Monsoon, let's plot the monthly rainfall for Monsoon season
# Aggregate data by month and year
forecast_data['Month'] = forecast_data.index.month
forecast_data['Year'] = forecast_data.index.year
monthly_data = forecast_data.groupby(['Year', 'Month']).agg({'PRECTOTCORR': 'sum', 'Predicted Precipitation (mm)': 'sum'})

# Filter for monsoon months (June to October)
monsoon_data = monthly_data.loc[(monthly_data.index.get_level_values(1) >= 6) & (monthly_data.index.get_level_values(1) <= 10)]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # Width of the bars

# Creating bars for actual and predicted data
ax.bar(monsoon_data.index - width/2, monsoon_data['PRECTOTCORR'], width, label='Actual')
ax.bar(monsoon_data.index + width/2, monsoon_data['Predicted Precipitation (mm)'], width, label='Predicted')

# Formatting the plot
ax.set_xlabel('Year, Month')
ax.set_ylabel('Total Precipitation (mm)')
ax.set_title('Monsoon Rainfall: Actual vs Predicted')
ax.legend()
ax.set_xticks(monsoon_data.index)
ax.set_xticklabels([f'{year}-{month}' for year, month in monsoon_data.index], rotation=45)

plt.tight_layout()
st.pyplot(fig)