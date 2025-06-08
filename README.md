# Chicago-2023-Bike-Climate


# Analyzing the Effect of Rain and Wind on Chicago's Divvy Bike-Share System

## Overview

This project investigates the impact of weather conditions—specifically precipitation and wind—on the demand for bike-sharing services in Chicago. Using 2023 data from the Divvy bike-share system and hourly meteorological records, this analysis employs a fixed-effects regression model to determine the causal effect of rain on the number of bike trips.

Additionally, the project explores the relationship between wind speed and direction and cyclists' travel patterns, examining whether riders prefer to travel with a tailwind versus a headwind.

The key methodologies used include geospatial analysis to merge trip data with the nearest weather station, panel data construction, and a two-way fixed-effects econometric model.

## Data Sources

  * **Bike-Share Data:** 2023 monthly trip data for the Chicago Divvy system. Sourced from individual CSV files (e.g., `202301-divvy-tripdata.csv`).
  * **Climate Data:** 2023 hourly meteorological data for the Chicago area, including precipitation, wind speed, and wind direction.

## Methodology

The analysis is conducted in several key stages within the Jupyter Notebook:

### 1\. Data Processing and Cleaning

  * **Bike & Climate Data Consolidation:** Functions `concat_all_bikes_data` and `concat_climate_data` read and merge the monthly CSV files from the `bike_trips/` and `Climate_DATA_2023/` directories, respectively.
  * **Trip Data Cleaning:** Trips are cleaned to remove outliers, such as those with excessively long durations (over 48 hours), and trips with missing start/end coordinates.
  * **Geospatial Station Recovery:** For trips with missing start or end station names, a geospatial nearest-neighbor join (`gpd.sjoin_nearest`) is used to assign the closest known station based on the trip's latitude and longitude.

### 2\. Geospatial Merging

To link each bike trip to its nearest weather conditions, the following steps were taken:

1.  Trip start times were floored to the nearest hour to create a time key.
2.  The Haversine formula was used to calculate the great-circle distance between each trip's starting point and all active meteorological stations for that hour.
3.  Each trip was then matched with the single closest meteorological station.

### 3\. Econometric Analysis: The Effect of Rain

A **two-way fixed-effects model** (`PanelOLS`) was used to isolate the causal effect of rain on bike trip volume. This approach controls for unobserved variables that are constant for each station (e.g., location popularity) and for factors that are constant across all stations at a specific time (e.g., city-wide events, seasonality).

  * **Dependent Variable:** `trip_count` (number of trips starting near a station in a given hour).
  * **Treatment Variable:** `treated` (a binary variable equal to 1 if there was any precipitation, 0 otherwise).
  * **Fixed Effects:** Station fixed effects and time (date-hour) fixed effects.

### 4\. Wind Analysis

The project also analyzes how wind affects travel behavior:

1.  **Travel Bearing:** The angle of travel (bearing) is calculated for each trip from its start and end coordinates.
2.  **Wind-Travel Difference:** The absolute difference between the wind direction and the travel bearing is calculated to determine if a trip was primarily with a tailwind (approx. 0°), headwind (approx. 180°), or crosswind (approx. 90°).
3.  **Visualization:** Polar plots are generated to show trip frequency based on this wind-travel difference across various wind speed categories (0-10 mph, 10-20 mph, etc.).

## Key Findings

  * **Effect of Rain:** The fixed-effects model indicates that the presence of rain has a statistically insignificant negative effect on the number of bike trips. The point estimate suggests a reduction of approximately **77 trips** per station-hour, but with a p-value of **0.195**, this result is not significant at conventional levels.
  * **Effect of Rain Intensity:** The intensity of rain (in inches per hour) also has a negative, though statistically insignificant (p-value: 0.173), effect on trip counts.
  * **Effect of Wind:** The polar plots suggest a behavioral pattern where cyclists tend to avoid riding directly into a headwind, a tendency that becomes more pronounced as wind speed increases.

## Visualizations

The notebook generates several visualizations to support the analysis:

  * **Heatmap:** A `folium` heatmap of Chicago showing the density of Divvy trip start/end points, with markers for the locations of meteorological stations.
  * **Fixed-Effects Plots:** Bar charts illustrating the demeaned (FE-adjusted) treatment effect of rain on trip counts, both overall and by the hour of the day.
  * **Wind-Travel Plots:** A series of polar plots showing trip frequency relative to wind direction for different wind speed categories.

*Left: Treatment effect with Fixed Effects. Center: Rain effect by hour. Right: Dose-response to rain intensity.*

*Trip frequency patterns based on wind-travel direction for different wind speed categories.*

## How to Run

1.  **Dependencies:** Ensure you have the required Python libraries installed:

    ```bash
    pip install pandas numpy geopandas statsmodels linearmodels folium matplotlib seaborn scikit-learn
    ```

2.  **File Structure:** Organize your files as follows:

    ```
    .
    ├── Project3_code_orig.ipynb
    ├── README.md
    ├── bike_trips/
    │   ├── 202301-divvy-tripdata.csv
    │   └── ... (other monthly trip data files)
    └── Climate_DATA_2023/
        └── ... (hourly climate data files)
    ```

3.  **Execution:** Run the cells in the `Project3_code_orig.ipynb` notebook sequentially. The notebook will process the data, perform the analysis, and generate the visualizations and an interactive map file (`chicago_bike_heatmap_example.html`).


Contact information:
Anton Liutin 
UW-Madison
Email and Phony by request