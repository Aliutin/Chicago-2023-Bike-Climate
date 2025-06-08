# Data Manipulation
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Geospatial Analysis
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

# Plotting and Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# Statistical Modeling
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS



def process_bike_data(path, geo_analysis=False, duration_threshold=2880):
    """
    Process bike trip data with optional geographic station recovery.
    
    Args:
        path (str): Path to the CSV file
        geo_analysis (bool): Whether to recover missing stations using spatial join
    
    Returns:
        pd.DataFrame: Processed bike data with duration and station information
    """
    # Read and process basic data
    bikes_df = pd.read_csv(path)
    bikes_df['started_at'] = pd.to_datetime(bikes_df['started_at'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    bikes_df['ended_at'] = pd.to_datetime(bikes_df['ended_at'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    bikes_df['duration'] = (bikes_df['ended_at'] - bikes_df['started_at']).dt.total_seconds() / 60


    ##################
    ##CLEANING DATA##
    ##################
    
    # Filter out trips longer than "duration_threshold" minutes (default 2 days)
    bikes_df = bikes_df[bikes_df['duration'] < duration_threshold]
    bikes_df = bikes_df[bikes_df['duration'] > 0]  # Ensure positive durations
    bikes_df = bikes_df[~((bikes_df['duration'] >1499) & (bikes_df['duration'] <1501))]  # Ensure positive durations
    # Drop rows with missing lat/lng for start and end stations
    bikes_df= bikes_df.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])


    if geo_analysis==False:
        return bikes_df
    else:
        ###################################################
        ###GEO ANALYSIS FOR RECOVERING MISSING STATIONS###
        ###################################################
        # Create station location data
        start_stations = bikes_df.groupby('start_station_name')[['start_lat', 'start_lng']].mean().reset_index()
        end_stations = bikes_df.groupby('end_station_name')[['end_lat', 'end_lng']].mean().reset_index()
        
        all_stations = pd.concat([
            start_stations.rename(columns={'start_station_name': 'station_name', 'start_lat': 'lat', 'start_lng': 'lng'}),
            end_stations.rename(columns={'end_station_name': 'station_name', 'end_lat': 'lat', 'end_lng': 'lng'})
        ]).groupby('station_name')[['lat', 'lng']].mean()
        
        # Geographic analysis for missing stations
    
    
        
        print("Creating station GeoDataFrame...")
        stations_gdf = gpd.GeoDataFrame(
            all_stations,
            geometry=gpd.points_from_xy(all_stations.lng, all_stations.lat),
            crs='EPSG:4326'
        ).to_crs('EPSG:3857').reset_index().drop_duplicates(subset=['geometry'])
        
        # Initialize recovery columns
        bikes_df['start_station_name_recovered'] = bikes_df['start_station_name'].copy()
        bikes_df['start_distance_to_station'] = 0.0
        bikes_df['end_station_name_recovered'] = bikes_df['end_station_name'].copy()
        bikes_df['end_distance_to_station'] = 0.0
        
        # Process missing start stations
        missing_start_mask = bikes_df['start_station_name'].isna()
        print(f"Missing start stations: {missing_start_mask.sum()}")
        
        if missing_start_mask.any():
            print("Processing missing start stations...")
            missing_start_gdf = gpd.GeoDataFrame(
                bikes_df[missing_start_mask],
                geometry=gpd.points_from_xy(bikes_df[missing_start_mask].start_lng, bikes_df[missing_start_mask].start_lat),
                crs='EPSG:4326'
            ).to_crs(stations_gdf.crs)
            
            joined_start = gpd.sjoin_nearest(
                missing_start_gdf, 
                stations_gdf[['geometry', 'station_name']], 
                how='left',
                distance_col='start_distance_to_station'
            ).reset_index().drop_duplicates(subset=['index'], keep='first').set_index('index')
            
            bikes_df.loc[missing_start_mask, 'start_station_name_recovered'] = joined_start['station_name'].values
            bikes_df.loc[missing_start_mask, 'start_distance_to_station'] = joined_start['start_distance_to_station'].values
        
        # Process missing end stations
        missing_end_mask = bikes_df['end_station_name'].isna()
        print(f"Missing end stations: {missing_end_mask.sum()}")
        
        if missing_end_mask.any():
            print("Processing missing end stations...")
            missing_end_gdf = gpd.GeoDataFrame(
                bikes_df[missing_end_mask],
                geometry=gpd.points_from_xy(bikes_df[missing_end_mask].end_lng, bikes_df[missing_end_mask].end_lat),
                crs='EPSG:4326'
            ).to_crs(stations_gdf.crs)
            
            joined_end = gpd.sjoin_nearest(
                missing_end_gdf, 
                stations_gdf[['geometry', 'station_name']], 
                how='left',
                distance_col='end_distance_to_station'
            ).reset_index().drop_duplicates(subset=['index'], keep='first').set_index('index')
            
            bikes_df.loc[missing_end_mask, 'end_station_name_recovered'] = joined_end['station_name'].values
            bikes_df.loc[missing_end_mask, 'end_distance_to_station'] = joined_end['end_distance_to_station'].values
        
        print("Station assignment complete!")
    
    return bikes_df


def concat_all_bikes_data(path_to_folder, geo_analysis=False, duration_threshold=2880):
    """
    Concatenate all bike data files in a folder and process them.
    
    Args:
        path_to_folder (str): Path to the folder containing bike data CSV files
        geo_analysis (bool): Whether to perform geographic station recovery
        duration_threshold (int): Maximum trip duration in minutes to consider valid
    
    Returns:
        pd.DataFrame: Concatenated and processed bike data
    """
    all_files = [f for f in os.listdir(path_to_folder) if f.endswith('.csv')]
    all_bikes_data = []
    
    for file in all_files:
        file_path = os.path.join(path_to_folder, file)
        print(f"Processing file: {file_path}")
        bikes_data = process_bike_data(file_path, geo_analysis, duration_threshold)
        all_bikes_data.append(bikes_data)
    
    return pd.concat(all_bikes_data, ignore_index=True)

def concat_climate_data(path_to_folder: str):
    all_climate = [f for f in os.listdir(path_to_folder) if f.endswith('.csv')]
    all_climate_data = []
    
    for file in all_climate:  # Fixed: was all_files
        file_path = os.path.join(path_to_folder, file)

        climate_data = pd.read_csv(file_path,low_memory=False)
        all_climate_data.append(climate_data)
        
    return pd.concat(all_climate_data, ignore_index=True)




import pandas as pd
import numpy as np

def merge_and_prepare_data(bike_df_full, climate_df_full):
    """
    Merges bike trip data with the closest meteorological station data.

    This function takes raw bike trip and climate data, merges them based on the nearest
    meteorological station at the time of each trip. It returns two dataframes:
    1. A trip-level dataframe with weather data appended to each trip.
    2. A panel dataset structured by station, date, and hour for aggregated analysis.

    Args:
        bike_df_full (pd.DataFrame): DataFrame containing bike trip data, including 'started_at',
                                     'start_lat', 'start_lng', and a unique 'ride_id'.
        climate_df_full (pd.DataFrame): DataFrame containing climate data, including 'DATE',
                                        'LATITUDE', 'LONGITUDE', and 'HourlyPrecipitation'.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - final_df (pd.DataFrame): The trip-level data, where each row is a single
                 bike trip matched with the data from the nearest weather station.
               - panel_data (pd.DataFrame): The aggregated panel dataset with trip counts
                 and weather information for each station, date, and hour.
    """
    print("Step 1: Converting datetime columns and creating time keys...")
    # 1. Convert to datetime objects and create a common time key
    bike_df_full['started_at'] = pd.to_datetime(bike_df_full['started_at'])
    climate_df_full['observation_time'] = pd.to_datetime(climate_df_full['DATE']).dt.floor('h')

    # 2. Create the common time key by flooring the trip's start time to the nearest hour.
    bike_df_full['time_key'] = bike_df_full['started_at'].dt.floor('h')


    # 3. Perform a Standard Merge on the Time Key ---
    # This matches each trip with ALL stations active in that hour.
    merged_df = pd.merge(
        bike_df_full,
        climate_df_full,
        left_on='time_key',
        right_on='observation_time',
        how='left'
    )


    # 4. Calculate Distance and Find the Nearest Station ---
    # Haversine function to calculate distance in kilometers
    def haversine_np(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km

    # Calculate the distance for every potential trip-station match
    merged_df['distance_km'] = haversine_np(
        merged_df['start_lng'],
        merged_df['start_lat'],
        merged_df['LONGITUDE'],
        merged_df['LATITUDE']
    )

    # For each original trip, find the index of the row with the smallest distance
    idx = merged_df.groupby('ride_id')['distance_km'].idxmin()

    # Select only the rows with the minimum distance to get the final, correct result
    final_df = merged_df.loc[idx].reset_index(drop=True)


    # Prepare the data with proper panel structure
    final_df['station_id']=final_df['LATITUDE'].astype(str)+final_df['LONGITUDE'].astype(str)
    final_df['hour'] = pd.to_datetime(final_df['started_at']).dt.hour
    final_df['date'] = pd.to_datetime(final_df['started_at']).dt.date
    final_df['date_hour'] = final_df['date'].astype(str) + '_' + final_df['hour'].astype(str)
    final_df['day_of_week'] = pd.to_datetime(final_df['date']).dt.day_name()

    # Create treatment variable
    final_df['treated'] = (final_df['HourlyPrecipitation'] > 0).astype(int)
    final_df['rain_intensity'] = final_df['HourlyPrecipitation']

    # Aggregate trips by station-date-hour
    trip_counts = final_df.groupby(['station_id', 'date', 'hour', 'date_hour']).agg({
        'treated': 'first',  # Rain status for that station-hour
        'rain_intensity': 'first',
        'started_at': 'count'  # Number of trips
    }).rename(columns={'started_at': 'trip_count'}).reset_index()

    # Create panel data structure - including stations with zero trips
    all_combinations = pd.MultiIndex.from_product([
        trip_counts['station_id'].unique(),
        trip_counts['date'].unique(),
        range(24)
    ], names=['station_id', 'date', 'hour']).to_frame(index=False)

    all_combinations['date_hour'] = all_combinations['date'].astype(str) + '_' + all_combinations['hour'].astype(str)

    # Merge to get complete panel
    panel_data = all_combinations.merge(
        trip_counts, 
        on=['station_id', 'date', 'hour', 'date_hour'], 
        how='left'
    )
    panel_data['trip_count'] = panel_data['trip_count'].fillna(0)
    panel_data['treated'] = panel_data['treated'].fillna(0)
    panel_data['rain_intensity'] = panel_data['rain_intensity'].fillna(0)


    return final_df, panel_data


def create_bike_heatmap(
    bike_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    map_center: list[float] = [41.8781, -87.6298],
    zoom_start: int = 11,
    sample_fraction: float = 0.1,
    heatmap_radius: int = 15,
    heatmap_blur: int = 10,
    output_filename: str = "chicago_bike_heatmap.html"
) -> folium.Map:
    """
    Generates and saves a Folium map with a heatmap of bike trip start/end points
    and markers for meteorological stations.

    Args:
        bike_df (pd.DataFrame): DataFrame containing bike trip data with columns
                                'start_lat', 'start_lng', 'end_lat', 'end_lng'.
        climate_df (pd.DataFrame): DataFrame containing climate data with columns
                                   'LATITUDE' and 'LONGITUDE' for station locations.
        map_center (List[float]): The latitude and longitude to center the map on.
                                  Defaults to Chicago's coordinates.
        zoom_start (int): The initial zoom level for the map. Defaults to 11.
        sample_fraction (float): The fraction of bike data to sample for the heatmap.
                                 Defaults to 0.1 (10%).
        heatmap_radius (int): The radius of influence for each point in the heatmap.
        heatmap_blur (int): The amount of blur to apply to the heatmap.
        output_filename (str): The name of the HTML file to save the map to.
                               Defaults to "chicago_bike_heatmap.html".

    Returns:
        folium.Map: The generated Folium map object.
    """
    # 1. Prepare the data
    # Extract unique meteorological station locations
    station_locations = climate_df[['LATITUDE', 'LONGITUDE']].drop_duplicates()

    # Sample the bike trip data for performance
    if sample_fraction < 1.0:
        bike_df_sampled = bike_df.sample(frac=sample_fraction, random_state=42)
    else:
        bike_df_sampled = bike_df

    # Extract bike trip start and end coordinates from the sampled data
    start_coords = bike_df_sampled[['start_lat', 'start_lng']].dropna().values.tolist()
    end_coords = bike_df_sampled[['end_lat', 'end_lng']].dropna().values.tolist()
    heat_data = start_coords + end_coords

    # 2. Create the map
    map_viz = folium.Map(location=map_center, zoom_start=zoom_start)

    # 3. Add the heatmap layer
    HeatMap(
        heat_data,
        radius=heatmap_radius,
        blur=heatmap_blur,
        max_zoom=13
    ).add_to(map_viz)

    # 4. Add markers for meteorological stations
    for _, row in station_locations.iterrows():
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=10,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"Met Station: ({row['LATITUDE']:.4f}, {row['LONGITUDE']:.4f})"
        ).add_to(map_viz)

    # 5. Save the map to an HTML file
    map_viz.save(output_filename)
    print(f"Map saved to {output_filename}")

    return map_viz