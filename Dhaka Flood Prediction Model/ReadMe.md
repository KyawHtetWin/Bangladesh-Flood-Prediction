# Machine Learning for Flood Prediction in Dhaka

As a key member of the model development team, I focused on data exploration, innovative feature engineering, and the development of predictive models, experimenting and incorporating topological and meteorological data to anticipate flood occurrences.

## Data Analysis and Feature Engineering

### Topological Features
- Analyzed Digital Elevation Model (DEM) and Slope data, consulting with domain experts to address outliers and validate data quality.

### Weather Features
- Examined the influence of weather variables like precipitation, temperature, and wind on flooding.
- Devised comprehensive feature engineering to encapsulate weather dynamics:
  - **Wind Vectorization**: Converted Wind Speed & Direction into x and y components.
  - **Cyclical Time Encoding**: Applied sine and cosine transformations to the 'day of the year' feature.
  - **Precipitation Analysis**: Developed lag features to quantify precipitation patterns, noting elevated averages on days with floods.
  - **Soil Saturation Trends**: Calculated changes in soil saturation over preceding days to assess flood risk.

## Model Development: Two separate models

### Spatial & Weather Condition Model
- The first model focuses on utilizing topographical & weather-related features of various locations in Dhaka city.
- Managed geographical data splits meticulously, resulting in a Random Forest model that emphasized recall and Cohen Kappa Score.
- Conclusions:
  - Removal of latitude & longitude from features led to metrics that could better generalize.
  - Topographical features, like Elevation, were identified as strong predictors through model feature importance.
  - Random Forest outperformed XGBoost, as evidenced by higher Cohen Kappa Score & ROC at a 0.4 threshold.

### Temporal Model
- The second model focuses on time-series modeling of two particular locations inside Dhaka city. 
- Best results achieved with an XGBoost model considering temporal factors:
  - High precision, balanced recall, low false positives/negatives.
  - Exceptional AUC of 0.99, indicating superior class separation ability.
  - Day of the year and precipitation were less significant predictors.
  - High F-1 and Cohen Kappa Scores, reflecting robust predictive accuracy.

## Future Work and Methodology Refinement
- Collaborating with a colleague to refine our data collection methodology, which currently employs random sampling of locations within Dhaka based on historical flood data.
- This approach designates certain points as perennially flooded or not, throughout the time-series dataset.
- We are working to enhance this methodology, recognizing that my experiments serve as a progressive step towards a more refined data collection strategy.

## Conclusions
Our work has demonstrated a potential predictive system for flood events in Dhaka. The models and feature engineering techniques developed here significantly advance disaster management and predictive analytics capabilities.

