import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model and test data
model = joblib.load('final_xgboost_model.joblib')
test_data = pd.read_csv('test_data.csv', index_col='Date', parse_dates=True)

# Generate predictions
y_pred = model.predict(test_data.drop(columns=['PRECTOTCORR']))

# Streamlit app layout based on the sketch
st.title('Rainfall Prediction in Rangpur')

# Placeholder for image of Rangpur
st.image('Rangpur_1.jpg', caption='Rangpur')  # Replace with your image path

st.header('Motivation & Goals')
st.write('Here you describe the motivation and the goals of this project.')

st.header('Model & Results')
st.write('Here you provide details about the model and the results.')

st.header('Key Findings')
st.write('Here you summarize the key findings from your model.')

st.header('Conclusion')
st.write('Here you write your conclusion and any insights.')

# Display the table with predicted values
st.subheader('Forecast Table')

# Assuming 'y_pred' contains the predicted values for the test set
# and 'X_test' is the DataFrame containing the test features with 'Date' as its index

# Convert the index to a column to display it as part of the table
forecast_data = test_data.copy()
forecast_data['Predicted Precipitation (mm)'] = y_pred

# Sort the DataFrame by the index (which is the Date) in descending order
forecast_data.sort_index(ascending=False, inplace=True)

# Use the slider to determine the number of forecast days to display
forecast_days = st.slider('Number of Forecast Days', 1, len(forecast_data), 30)

# Select the top 'forecast_days' entries after sorting
forecast_display = forecast_data.head(forecast_days)[['Predicted Precipitation (mm)']]

# Reset the index to turn the Date index into a column for displaying
forecast_display.reset_index(inplace=True)

# Number of days for forecasting
st.write(f"Forecasting for most recent {forecast_days} days")

# Display the DataFrame in the Streamlit app
st.write(forecast_display)

# Plot comparing predicted and actual precipitation
st.subheader('Precipitation Forecast vs. Actual Plot')
fig, ax = plt.subplots()
ax.plot(forecast_data.index, forecast_data['PRECTOTCORR'], label='Actual', marker='o', color='blue')
ax.plot(forecast_data.index, forecast_data['Predicted Precipitation (mm)'], label='Predicted', marker='x', color='red', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Precipitation (mm)')
ax.legend()
st.pyplot(fig)

