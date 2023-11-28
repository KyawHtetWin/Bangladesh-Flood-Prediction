import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the model and test data
model = joblib.load('final_xgboost_model.joblib')
test_data = pd.read_csv('test_data.csv', index_col='Date', parse_dates=True)

# Generate predictions
y_pred = model.predict(test_data.drop(columns=['PRECTOTCORR']))

# Streamlit app layout based on the sketch
st.title('Rainfall Prediction in Rangpur')

# Placeholder for image of Rangpur
st.image('Rangpur_1.jpg', caption='Tajhat Palace in Rangpur')  

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