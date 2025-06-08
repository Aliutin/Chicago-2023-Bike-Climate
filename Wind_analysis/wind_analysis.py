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


def calculate_bearing_and_distance(df):
    """
    Calculate the bearing (angle) and direct distance between start and end coordinates.
    
    Bearing: 0° = North, 90° = East, 180° = South, 270° = West (clockwise)
    Distance: Great circle distance in kilometers
    """
    
    # Convert degrees to radians
    start_lat_rad = np.radians(df['start_lat'])
    start_lng_rad = np.radians(df['start_lng'])
    end_lat_rad = np.radians(df['end_lat'])
    end_lng_rad = np.radians(df['end_lng'])
    
    # Calculate difference in longitude
    dlon = end_lng_rad - start_lng_rad
    
    # Calculate bearing (angle of travel)
    y = np.sin(dlon) * np.cos(end_lat_rad)
    x = (np.cos(start_lat_rad) * np.sin(end_lat_rad) - 
         np.sin(start_lat_rad) * np.cos(end_lat_rad) * np.cos(dlon))
    
    # Calculate bearing in radians, then convert to degrees
    bearing_rad = np.arctan2(y, x)
    bearing_deg = np.degrees(bearing_rad)
    
    # Normalize to 0-360 degrees (0° = North, clockwise)
    bearing_deg = (bearing_deg + 360) % 360
    
    # Calculate great circle distance using Haversine formula
    # Earth's radius in kilometers
    R = 6371.0
    
    # Haversine formula
    dlat = end_lat_rad - start_lat_rad
    a = (np.sin(dlat/2)**2 + 
         np.cos(start_lat_rad) * np.cos(end_lat_rad) * np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = R * c
    
    return bearing_deg, distance_km

def create_wind_trip_plots(analysis_df):
    """
    Create 3 half-circle plots in a row showing trip frequency 
    by wind-travel direction difference for different wind speed bins.
    """
    
    # Define wind speed bins
    wind_speed_bins = [(0, 10), (10,20), (20, 30), (30, float('inf'))]
    wind_speed_labels = ['0-10 mph', '10-20 mph', '20-30 mph', '30+ mph']
    
    # Create 10-degree bins for wind_travel_diff (0-180 degrees)
    bins = np.arange(0, 181, 10)
    bin_centers = bins[:-1] + 5
    
    # Create figure with 3 subplots in a row
    fig, axes = plt.subplots(1, len(wind_speed_bins), figsize=(4*len(wind_speed_bins), 4), subplot_kw=dict(projection='polar'))
    
    # Process each wind speed bin
    for i, ((min_speed, max_speed), label) in enumerate(zip(wind_speed_bins, wind_speed_labels)):
        ax = axes[i]
        
        # Filter data for current wind speed bin
        if max_speed == float('inf'):
            mask = analysis_df['HourlyWindSpeed'] >= min_speed
        else:
            mask = (analysis_df['HourlyWindSpeed'] >= min_speed) & (analysis_df['HourlyWindSpeed'] < max_speed)
        
        bin_data = analysis_df[mask]['wind_travel_diff']
        
        if len(bin_data) == 0:
            ax.set_title(f'{label}\n(No data)', fontsize=12)
            continue
        
        # Count trips in each 10-degree bin
        counts, _ = np.histogram(bin_data, bins=bins)
        total_trips = len(bin_data)
        
        # Normalize to percentage (where 100 = total_trips/18)
        max_percentage = 100
        normalized_counts = (counts / total_trips) * 18 * max_percentage if total_trips > 0 else counts
        
        # Convert to radians
        theta = np.deg2rad(bin_centers)
        width = np.deg2rad(10)
        
        # Create bars
        ax.bar(theta, normalized_counts, width=width, alpha=0.7)
        
        # Configure polar plot
        ax.set_theta_zero_location('N')  # 0° at top
        ax.set_theta_direction(1)  # Clockwise
        ax.set_thetalim(0, np.pi)  # Half circle
        ax.set_ylim(0, max_percentage)
        
        # Grid and ticks
        ax.grid(True, alpha=0.3)
        ax.set_thetagrids(np.arange(0, 181, 30))
        
        # Title
        ax.set_title(f'{label}\n({total_trips} trips)', fontsize=12)
    
    plt.tight_layout()
    fig.suptitle('Bike Trip Analysis: Wind-Travel Direction Patterns', 
                 fontsize=16, y=1.2)
    
    # Legend
    legend_text = [
        "Wind Speed Categories arranged into 4 categories",
        "Angles: 0° = tailwind, 90° = crosswind, 180° = headwind", 
        "Bar height: percentage of trips in each 10° direction bin",
        "Scale: 100% = total trips for this wind speed ÷ 18"
    ]
    
    fig.text(0.5, -0.25, '\n'.join(legend_text), fontsize=11, ha='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    plt.show()
    return fig


def prepare_wind_analysis_dataset(final_df, min_distance=0.250, max_distance=150, 
                                 save_csv=False, csv_filename='trip_wind_data.csv'):
    """
    Prepare bike trip dataset for wind analysis by calculating wind-travel interactions
    and filtering anomalous trips.
    
    Parameters:
    -----------
    final_df : pandas.DataFrame
        DataFrame containing bike trip data with columns: duration, HourlyWindDirection, 
        HourlyWindSpeed, direct_distance_km, travel_angle, date, hour
    min_distance : float, default=0.250
        Minimum trip distance in km to include
    max_distance : float, default=150
        Maximum trip distance in km to include
    save_csv : bool, default=False
        Whether to save the analysis dataset to CSV
    csv_filename : str, default='trip_wind_data.csv'
        Filename for saved CSV if save_csv is True
    
    Returns:
    --------
    pandas.DataFrame
        Processed analysis dataset with wind-travel interaction variables
    """
    
    def angle_difference(wind_dir, travel_dir):
        """Calculate the smallest angle difference between wind and travel direction"""
        diff = abs(wind_dir - travel_dir)
        return np.minimum(diff, 360 - diff)
    
    # Apply distance and direction calculations (assuming calculate_bearing_and_distance exists)
    df = final_df.copy()
    df['travel_angle'], df['direct_distance_km'] = calculate_bearing_and_distance(df)
    
    # Filter anomalous travel distances
    df = df[df['direct_distance_km'] < max_distance]
    df = df[df['direct_distance_km'] > min_distance]
    
    # Prepare analysis dataset with required columns
    required_cols = ['duration', 'HourlyWindDirection', 'HourlyWindSpeed', 
                    'direct_distance_km', 'travel_angle', 'date', 'hour']
    analysis_df = df[required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    print(f"Analysis dataset shape: {analysis_df.shape}")
    
    # Create wind-travel interaction variables
    analysis_df['wind_travel_diff'] = angle_difference(
        analysis_df['HourlyWindDirection'], 
        analysis_df['travel_angle']
    )
    
    # Create wind direction components (unit vectors)
    analysis_df['wind_x'] = np.cos(np.radians(analysis_df['HourlyWindDirection']))
    analysis_df['wind_y'] = np.sin(np.radians(analysis_df['HourlyWindDirection']))
    
    # Create travel direction components (unit vectors)
    analysis_df['travel_x'] = np.cos(np.radians(analysis_df['travel_angle']))
    analysis_df['travel_y'] = np.sin(np.radians(analysis_df['travel_angle']))
    
    # Wind-travel alignment (dot product of unit vectors)
    # Positive = tailwind, Negative = headwind, Zero = crosswind
    analysis_df['wind_alignment'] = (
        analysis_df['wind_x'] * analysis_df['travel_x'] + 
        analysis_df['wind_y'] * analysis_df['travel_y']
    )
    
    # Convert date to datetime and create temporal features
    analysis_df['date'] = pd.to_datetime(analysis_df['date'])
    analysis_df['day_of_week'] = analysis_df['date'].dt.dayofweek
    analysis_df['month'] = analysis_df['date'].dt.month
    
    # Save to CSV if requested
    if save_csv:
        analysis_df.to_csv(csv_filename, index=False)
        print(f"Analysis dataset saved to {csv_filename}")
    
    return analysis_df