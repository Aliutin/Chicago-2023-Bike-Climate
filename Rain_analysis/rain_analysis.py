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


def analyze_rain_treatment_effect(
    panel_data,
    trip_col='trip_count',
    treatment_col='treated',
    intensity_col='rain_intensity',
    station_col='station_id',
    datetime_col='date_hour',
    date_col='date',
    hour_col='hour',
    cluster_se=True,
    intensity_bins=None,
    figsize=(18, 6),
    show_plots=True
):
    """
    Analyze the treatment effect of rain on trip counts using fixed effects regression.
    
    Parameters
    ----------
    panel_data : pd.DataFrame
        Panel data with trip counts, treatment indicator, and other variables
    trip_col : str, default='trip_count'
        Column name for trip counts
    treatment_col : str, default='treated'
        Column name for treatment indicator (1=rain, 0=no rain)
    intensity_col : str, default='rain_intensity'
        Column name for rain intensity
    station_col : str, default='station_id'
        Column name for station identifier
    datetime_col : str, default='date_hour'
        Column name for date-hour timestamp
    date_col : str, default='date'
        Column name for date
    hour_col : str, default='hour'
        Column name for hour of day
    cluster_se : bool, default=True
        Whether to use clustered standard errors
    intensity_bins : list, optional
        Custom bins for rain intensity analysis. 
        Default: [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
    figsize : tuple, default=(18, 6)
        Figure size for plots
    show_plots : bool, default=True
        Whether to display plots
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'main_results': Main regression results
        - 'intensity_results': Rain intensity regression results
        - 'hourly_effects': Hourly treatment effects
        - 'cleaned_data': Cleaned panel data with FE adjustments
        - 'summary_stats': Summary statistics
    """
    
    # ============================================================================
    # DATA PREPARATION
    # ============================================================================
    
    # Create a copy to avoid modifying original data
    panel_data_clean = panel_data.copy()
    
    # Clean data and ensure proper types
    required_cols = [trip_col, treatment_col, intensity_col, station_col, datetime_col]
    panel_data_clean = panel_data_clean.dropna(subset=required_cols)
    
    # Convert to numeric
    for col in [trip_col, treatment_col, intensity_col]:
        panel_data_clean[col] = pd.to_numeric(panel_data_clean[col], errors='coerce')
    
    # Create time variables
    panel_data_clean['week'] = pd.to_datetime(panel_data_clean[date_col]).dt.isocalendar().week
    panel_data_clean['day_of_week'] = pd.to_datetime(panel_data_clean[date_col]).dt.day_name()
    
    # Create numeric time index for linearmodels
    le_time = LabelEncoder()
    panel_data_clean['time_numeric'] = le_time.fit_transform(panel_data_clean[datetime_col])
    
    # Summary statistics
    summary_stats = {
        'n_observations': panel_data_clean.shape[0],
        'n_stations': panel_data_clean[station_col].nunique(),
        'n_time_periods': panel_data_clean[datetime_col].nunique(),
        'pct_treated': (panel_data_clean[treatment_col] == 1).mean() * 100
    }
    
    print(f"Clean dataset: {summary_stats['n_observations']} observations")
    print(f"Unique stations: {summary_stats['n_stations']}")
    print(f"Unique date-hours: {summary_stats['n_time_periods']}")
    print(f"Percentage treated (rain): {summary_stats['pct_treated']:.1f}%")
    
    # ============================================================================
    # FIXED EFFECTS REGRESSION
    # ============================================================================
    
    # Set proper panel index with numeric time
    panel_indexed = panel_data_clean.set_index([station_col, 'time_numeric'])
    
    # Main regression: Rain treatment effect with two-way fixed effects
    fe_model = PanelOLS(
        dependent=panel_indexed[trip_col],
        exog=panel_indexed[[treatment_col]],
        entity_effects=True,   # Station fixed effects
        time_effects=True      # Date-hour fixed effects
    ).fit(cov_type='clustered' if cluster_se else 'robust', cluster_entity=cluster_se)
    
    main_results = {
        'coefficient': fe_model.params[treatment_col],
        'std_error': fe_model.std_errors[treatment_col],
        'p_value': fe_model.pvalues[treatment_col],
        'conf_int_lower': fe_model.conf_int().loc[treatment_col, 'lower'],
        'conf_int_upper': fe_model.conf_int().loc[treatment_col, 'upper'],
        'model': fe_model
    }
    
    print("\n" + "="*60)
    print("MAIN RESULTS: Rain Treatment Effect")
    print("="*60)
    print(f"Treatment effect (rain): {main_results['coefficient']:.3f} trips")
    print(f"Standard error: {main_results['std_error']:.3f}")
    print(f"P-value: {main_results['p_value']:.4f}")
    print(f"95% CI: [{main_results['conf_int_lower']:.3f}, {main_results['conf_int_upper']:.3f}]")
    
    # Rain intensity regression
    fe_model_intensity = PanelOLS(
        dependent=panel_indexed[trip_col],
        exog=panel_indexed[[intensity_col]],
        entity_effects=True,
        time_effects=True
    ).fit(cov_type='clustered' if cluster_se else 'robust', cluster_entity=cluster_se)
    
    intensity_results = {
        'coefficient': fe_model_intensity.params[intensity_col],
        'std_error': fe_model_intensity.std_errors[intensity_col],
        'p_value': fe_model_intensity.pvalues[intensity_col],
        'model': fe_model_intensity
    }
    
    print(f"\nRain intensity effect: {intensity_results['coefficient']:.3f} trips per inch")
    print(f"P-value: {intensity_results['p_value']:.4f}")
    
    # ============================================================================
    # COMPUTE FIXED EFFECTS ADJUSTMENTS
    # ============================================================================
    
    # Demean by station and time fixed effects
    panel_data_clean['trip_count_demeaned'] = panel_data_clean[trip_col].copy()
    
    # Remove station fixed effects (station means)
    station_means = panel_data_clean.groupby([station_col, 'week'])[trip_col].transform('mean')
    panel_data_clean['trip_count_demeaned'] -= station_means
    
    # Remove time fixed effects (date-hour means)
    time_means = panel_data_clean.groupby([hour_col, 'day_of_week'])['trip_count_demeaned'].transform('mean')
    panel_data_clean['trip_count_demeaned'] -= time_means
    
    # Compute hourly effects
    hourly_effects = []
    panel_data_clean['trip_fe'] = np.nan
    
    for hour in range(24):
        hour_data = panel_data_clean[panel_data_clean[hour_col] == hour].copy()
        if len(hour_data) > 0:
            # Remove station-week FE
            hour_station_means = hour_data.groupby([station_col, 'week'])[trip_col].transform('mean')
            hour_data['trip_fe'] = hour_data[trip_col] - hour_station_means
            
            # Remove hour-day_of_week FE
            hour_date_means = hour_data.groupby([hour_col, 'day_of_week'])['trip_fe'].transform('mean')
            hour_data['trip_fe'] -= hour_date_means
            
            # Update main panel
            panel_data_clean.loc[panel_data_clean[hour_col] == hour, 'trip_fe'] = hour_data['trip_fe'].values
            
            # Calculate treatment effect
            treated_fe = hour_data[hour_data[treatment_col] == 1]['trip_fe'].mean()
            control_fe = hour_data[hour_data[treatment_col] == 0]['trip_fe'].mean()
            
            if not (pd.isna(treated_fe) or pd.isna(control_fe) or 
                    len(hour_data[hour_data[treatment_col] == 1]) < 10):
                hourly_effects.append({
                    'hour': hour,
                    'effect': treated_fe - control_fe,
                    'n_treated': len(hour_data[hour_data[treatment_col] == 1])
                })
            else:
                hourly_effects.append({'hour': hour, 'effect': 0, 'n_treated': 0})
        else:
            hourly_effects.append({'hour': hour, 'effect': 0, 'n_treated': 0})
    
    # ============================================================================
    # VISUALIZATIONS
    # ============================================================================
    
    if show_plots:
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Treatment effect with fixed effects
        _plot_treatment_effect(axes[0], panel_data_clean, treatment_col)
        
        # Plot 2: Hourly treatment effects
        _plot_hourly_effects(axes[1], hourly_effects)
        
        # Plot 3: Rain intensity dose-response
        _plot_dose_response(axes[2], panel_data_clean, treatment_col, intensity_col, 
                          station_col, datetime_col, intensity_bins)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'main_results': main_results,
        'intensity_results': intensity_results,
        'hourly_effects': hourly_effects,
        'cleaned_data': panel_data_clean,
        'summary_stats': summary_stats
    }


def _plot_treatment_effect(ax, data, treatment_col):
    """Plot treatment effect with fixed effects."""
    treated_mean_fe = data[data[treatment_col] == 1]['trip_count_demeaned'].mean()
    control_mean_fe = data[data[treatment_col] == 0]['trip_count_demeaned'].mean()
    
    bars = ax.bar(['No Rain (FE)', 'Rain (FE)'], [control_mean_fe, treated_mean_fe],
                   color=['skyblue', 'salmon'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('FE-Adjusted Trip Count')
    ax.set_title('Treatment Effect with Station & Time Fixed Effects')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.5),
                f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold')


def _plot_hourly_effects(ax, hourly_effects):
    """Plot hourly treatment effects."""
    hours = [h['hour'] for h in hourly_effects]
    effects = [h['effect'] for h in hourly_effects]
    colors = ['red' if x < 0 else 'green' for x in effects]
    
    ax.bar(hours, effects, color=colors, alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('FE-Adjusted Treatment Effect')
    ax.set_title('Rain Effect by Hour (with Fixed Effects)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3)


def _plot_dose_response(ax, data, treatment_col, intensity_col, station_col, 
                       datetime_col, intensity_bins=None):
    """Plot rain intensity dose-response."""
    if intensity_bins is None:
        intensity_bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
    
    rain_data = data[data[treatment_col] == 1].copy()
    
    if len(rain_data) > 0:
        # Create intensity bins
        labels = [f'{intensity_bins[i]}-{intensity_bins[i+1]}' 
                  for i in range(len(intensity_bins)-1)]
        labels[-1] = f'{intensity_bins[-2]}+'
        
        rain_data['intensity_bin'] = pd.cut(rain_data[intensity_col],
                                           bins=intensity_bins,
                                           labels=labels)
        
        dose_effects = []
        dose_labels = []
        
        for intensity_bin in rain_data['intensity_bin'].cat.categories:
            bin_data = rain_data[rain_data['intensity_bin'] == intensity_bin]
            
            if len(bin_data) > 20:  # Only if enough observations
                # Get matched control observations
                control_matches = []
                for _, row in bin_data.iterrows():
                    station_date_controls = data[
                        (data[station_col] != row[station_col]) &
                        (data[datetime_col] == row[datetime_col]) &
                        (data[treatment_col] == 0)
                    ]
                    if len(station_date_controls) > 0:
                        control_matches.append(station_date_controls['trip_fe'].mean())
                
                if len(control_matches) > 0:
                    treatment_effect = bin_data['trip_fe'].mean() - np.mean(control_matches)
                    dose_effects.append(treatment_effect)
                    dose_labels.append(intensity_bin)
        
        if dose_effects:
            x_pos = np.arange(len(dose_effects))
            bars = ax.bar(x_pos, dose_effects, alpha=0.7, color='darkblue')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(dose_labels, rotation=45)
            ax.set_xlabel('Rain Intensity (inches)')
            ax.set_ylabel('Treatment Effect vs Matched Controls')
            ax.set_title('Dose-Response: Rain Intensity (Station-Date Matched)')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3)


