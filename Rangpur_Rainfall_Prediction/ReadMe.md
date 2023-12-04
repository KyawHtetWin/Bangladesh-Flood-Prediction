# Machine Learning Projects for Average Rainfall Forecasting in Rangpur


As part of machine learning development team, I contributed to the creation and refinement of predictive models for rainfall forecasting, focusing on the integration of various data types and feature engineering to enhance predictive capabilities.

## Rainfall Prediction for Rangpur

### Motivation & Goal
The objective of this notebook is to forecast average daily rainfall in Rangpur, leveraging historical weather patterns to inform and improve flood preparedness strategies.

### Model Development & Results
- Implemented an XGBoost regression model with hyper-parameter tuning, achieving promising predictive performance.
- **Key Metrics on Test Data:**
  - **RMSE**: 5.2992 mm per day
  - **MAE**: 2.8471 mm per day — a critical metric for this analysis.
  - **R^2**: 72.98%, indicating a strong fit to the historical data.

### Key Findings
- The feature `precip_mean_1`, representing the previous day's rainfall, emerged as the strongest predictor for the next day's rainfall.

### Conclusion
Given the daily average precipitation of 6.66 mm and a standard deviation of 12.80 mm, the model demonstrates a reasonable predictive power, suitable for practical application.

### Future Work Suggestions
- Integrate additional data sources to capture extreme precipitation events and improve model robustness.
- Enhance the feature engineering process to better represent complex rainfall patterns.

### Deployment
- Deployed a Streamlit application to showcase predictions of rainfall for the year 2023 based on the test data, making the insights accessible to the public and relevant stakeholders.
