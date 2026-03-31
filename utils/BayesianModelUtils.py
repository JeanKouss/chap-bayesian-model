import pandas as pd
import numpy as np
import geopandas as gpd
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

# Create a custom directory for compiled files inside your project
# This avoids Windows AppLocker blocking the system TEMP folder
os.makedirs('pymc_cache', exist_ok=True)
os.environ['PYTENSOR_FLAGS'] = f"base_compiledir={os.path.abspath('pymc_cache')}"
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU if JAX is causing the DLL issue

warnings.filterwarnings('ignore')

def load_and_preprocess_training_data(df, geojson_path, covariates):
    # Load GeoJSON
    gdf = gpd.read_file(geojson_path)

    # Standardize location names
    df = df.copy()
    df['location_clean'] = df['location'].str.upper().str.strip()
    gdf['name_clean'] = gdf['name'].str.upper().str.strip()

    # Check for mismatches
    locations_df = set(df['location_clean'].unique())
    locations_geo = set(gdf['name_clean'].unique())
    
    common_locations = locations_df & locations_geo
    
    if not common_locations:
        raise ValueError("No common locations found between DataFrame and GeoJSON!")

    missing_in_geo = locations_df - locations_geo
    if missing_in_geo:
        print(f"locations in DataFrame but not in GeoJSON (dropped): {missing_in_geo}")
        
    missing_in_df = locations_geo - locations_df
    if missing_in_df:
        print(f"locations in GeoJSON but not in DataFrame (dropped): {missing_in_df}")

    # Keep only common locations
    df = df[df['location_clean'].isin(common_locations)]
    gdf = gdf[gdf['name_clean'].isin(common_locations)]

    # Convert month to datetime and sort
    df['time_period'] = pd.to_datetime(df['time_period'], format='%Y-%m')
    df = df.sort_values(['location_clean', 'time_period'])

    # Create indices
    locations = sorted(df['location_clean'].unique())
    dates = sorted(df['time_period'].unique())

    location_to_idx = {d: i for i, d in enumerate(locations)}
    date_to_idx = {d: i for i, d in enumerate(dates)}

    df['location_idx'] = df['location_clean'].map(location_to_idx)
    df['time_idx'] = df['time_period'].map(date_to_idx)

    # Temporal variables
    df['month'] = df['time_period'].dt.month
    df['year'] = df['time_period'].dt.year
    # df['time_linear'] = (df['time_period'] - df['time_period'].min()).dt.days / 365.25

    # Check covariates existence
    missing_covs = [cov for cov in covariates if cov not in df.columns]
    if missing_covs:
        print(f"Missing covariates: {missing_covs}")
        covariates = [cov for cov in covariates if cov in df.columns]

    # Scale covariates
    scaler = StandardScaler()
    df_scaled = df.copy()
    if covariates:
        df_scaled[covariates] = scaler.fit_transform(df[covariates])

    n_locations = len(locations)
    n_times = len(dates)
    n_obs = len(df)

    print(f"Data prepared: {n_locations} locations, {n_times} periods, {n_obs} observations")
    
    return {
        'df': df_scaled, # 'df': Processed DataFrame with indices and features.
        'gdf': gdf, # 'gdf': GeoDataFrame with location geometries.
        'location_to_idx': location_to_idx, # 'location_to_idx': Mapping from location names to integer indices.
        'training_date_to_idx': date_to_idx, # 'date_to_idx': Mapping from dates to integer indices.
        'covariates': covariates, # 'covariates': List of covariate names used in the model.
        'scaler': scaler, # 'scaler': Fitted StandardScaler for covariates.
        'n_locations': n_locations, # 'n_locations': Number of locations.
        'training_n_times': n_times, # 'n_times': Number of time periods.
        'n_training_obs': n_obs # 'n_obs': Number of observations.
    }


def compute_adjacency_matrix(gdf, location_to_idx):
    # Reorder GeoDataFrame to match location indices
    location_order = [d for d in location_to_idx.keys()]
    gdf_ordered = gdf.set_index('name_clean').reindex(location_order).reset_index()

    n = len(gdf_ordered)
    W = np.zeros((n, n))

    # Build adjacency based on 'touches'
    for i in range(n):
        for j in range(i+1, n):
            if gdf_ordered.geometry.iloc[i].touches(gdf_ordered.geometry.iloc[j]):
                W[i, j] = 1
                W[j, i] = 1

    # Fix isolated locations
    row_sums = W.sum(axis=1)
    isolated = np.where(row_sums == 0)[0]

    if len(isolated) > 0:
        print(f"Warning: {len(isolated)} isolated locations detected. Connecting to nearest neighbors.")
        
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf_ordered.geometry])

        for iso_idx in isolated:
            distances = cdist([centroids[iso_idx]], centroids)[0]
            distances[iso_idx] = np.inf  # Exclude self
            nearest_idx = np.argmin(distances)

            W[iso_idx, nearest_idx] = 1
            W[nearest_idx, iso_idx] = 1
            print(f"  location {iso_idx} connected to {nearest_idx}")

    # Ensure symmetry
    W = np.maximum(W, 0)
    W = (W + W.T) / 2
    
    # Calculate Degree matrix
    D = np.diag(W.sum(axis=1))
    
    print(f"Adjacency matrix created: {np.sum(W)/2:.0f} connections")
    return (
        W, # 'W': Adjacency matrix (n_locations x n_locations).
        D, # 'D': Degree matrix (diagonal matrix of row sums of W).
        gdf_ordered # 'gdf_ordered': GeoDataFrame reordered to match the indices.
    )


def build_malaria_model(training_df, covariates, n_locations, n_times, W, hyperparams=None):
    with pm.Model() as model:
        # --- Data Containers (Mutable for predictions) ---
        location_idx = pm.Data("location_idx", training_df['location_idx'].values)
        time_idx = pm.Data("time_idx", training_df['time_idx'].values)
        month = pm.Data("month", training_df['month'].values)
        
        # Covariates matrix

        X = pm.Data("X", training_df[covariates].values)
        
        # Observed data (Should be training data only, no NaNs)
        y_obs = pm.Data("y_obs", training_df['disease_cases'].values.astype('int32'))

        # --- Priors ---
        
        # 1. Global Intercept and Coefficients
        alpha = pm.Normal("alpha", mu=0, sigma=2.5)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=len(covariates))

        # 2. Spatial Component (CAR)
        # Precision of spatial effect
        tau_spatial = pm.Gamma("tau_spatial", alpha=1, beta=0.5)
        # Spatial autocorrelation (0 to 1)
        rho = pm.Beta("rho", alpha=1, beta=1)
        
        W_tensor = pt.as_tensor_variable(W.astype('float32'))
        
        # CAR Prior
        spatial_effect = pm.CAR(
            "spatial_effect",
            mu=np.zeros(n_locations, dtype='float32'),
            W=W_tensor,
            alpha=rho,
            tau=tau_spatial
        )

        # 3. Temporal Component (Random Walk)
        sigma_rw = pm.HalfNormal("sigma_rw", sigma=0.2)
        temporal_rw = pm.GaussianRandomWalk(
            "temporal_rw",
            sigma=sigma_rw,
            shape=n_times
        )

        # 4. Seasonal Component (12 months)
        sigma_seasonal = pm.HalfNormal("sigma_seasonal", sigma=0.3)
        seasonal_effect = pm.Normal(
            "seasonal_effect",
            mu=0, sigma=sigma_seasonal,
            shape=12
        )

        # 5. Spatio-temporal Interaction (Gaussian Random Walk per location)
        sigma_st = pm.HalfNormal("sigma_st", sigma=0.05)
        spatiotemporal_effect = pm.GaussianRandomWalk(
            "spatiotemporal_effect",
            sigma=sigma_st,
            shape=(n_locations, n_times)
        )

        # --- Linear Predictor ---
        linear_pred = alpha + pm.math.dot(X, beta)
        
        # Indexing effects
        spatial_contrib = spatial_effect[location_idx]
        temporal_contrib = temporal_rw[time_idx]
        seasonal_contrib = seasonal_effect[month - 1] # Month 1-12 -> Index 0-11
        st_contrib = spatiotemporal_effect[location_idx, time_idx]

        log_rate = (linear_pred + 
                   spatial_contrib + 
                   temporal_contrib + 
                   seasonal_contrib + 
                   st_contrib)

        # Inverse Link (Log)
        mu = pm.math.exp(log_rate)

        # --- Likelihood ---
        pm.Poisson("likelihood", mu=mu, observed=y_obs)

    print("Model built successfully!")
    return model

def train_model(model, draws=2000, tune=1000, chains=4, target_accept=0.95):
    print(f"Training model ({draws} draws, {tune} tune, {chains} chains)...")
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=42,
            return_inferencedata=True,
            nuts_sampler="numpyro" # Faster sampler if available
        )
    print("Training complete!")
    return trace # arviz.InferenceData : The trace containing posterior samples and stats.

def prepare_prediction_data(df_future, scaler, location_to_idx, date_to_idx, covariates):
    
    df_pred = df_future.copy()
    
    # Cleaning
    df_pred['location_clean'] = df_pred['location'].str.upper().str.strip()
    df_pred['time_period'] = pd.to_datetime(df_pred['time_period'], format='%Y-%m')
    
    # Check locations
    unknown_locations = set(df_pred['location_clean']) - set(location_to_idx.keys())
    if unknown_locations:
        raise ValueError(f"Unknown locations in future data: {unknown_locations}")
        
    df_pred['location_idx'] = df_pred['location_clean'].map(location_to_idx)
    
    # Handle Time Indices (Extension)
    # If time_period is known, use existing index. If new, increment max index.
    max_train_idx = max(date_to_idx.values())
    unique_pred_dates = sorted(df_pred['time_period'].unique())
    
    new_date_map = {}
    current_idx = max_train_idx + 1
    
    for time_period in unique_pred_dates:
        if time_period in date_to_idx:
            new_date_map[time_period] = date_to_idx[time_period]
        else:
            new_date_map[time_period] = current_idx
            current_idx += 1
            
    df_pred['time_idx'] = df_pred['time_period'].map(new_date_map)
    df_pred['month'] = df_pred['time_period'].dt.month
    
    # Covariates Scaling
    df_pred[covariates] = scaler.transform(df_pred[covariates])
    
    print(f"Prediction data prepared: {len(df_pred)} observations.")
    return df_pred # pandas.DataFrame : Processed future dataframe ready for the model.

def predict_malaria(model, covariates, trace, df_new):
    
    # If standard PyMC sampling (not NumPyro), predictions are straightforward
    with model:
        # We must pass the exact same shapes/types as defined in build_model
        # Note: pm.set_data must match the names given in pm.Data()
        pm.set_data({
            "location_idx": df_new['location_idx'].values.astype('int32'),
            "time_idx": df_new['time_idx'].values.astype('int32'),
            "month": df_new['month'].values.astype('int32'),
            "X": df_new[covariates].values.astype('float32'),
            "y_obs": np.zeros(len(df_new), dtype='int32') # Shape must match df_new
        })
        
        target_df = df_new.copy()
        
        # Sample from the posterior predictive
        ppc = pm.sample_posterior_predictive(
            trace,
            predictions=True, # Keeps the original trace separate if needed (depends on version)
            random_seed=42
        )
    
    # Extract prediction key (usually 'likelihood' or the observed node name)
    pred_samples = ppc.predictions['likelihood']
    # if hasattr(ppc, "predictions"):
    #     print('Using ppc.predictions for predictions')
    # elif hasattr(ppc, "posterior_predictive"):
    #     print('Using ppc.posterior_predictive for predictions')
    #     pred_samples = ppc.posterior_predictive['likelihood']
    # else:
    #     print('Using ppc["likelihood"] for predictions')
    #     pred_samples = ppc['likelihood']
    
    # Calculate summary statistics (mean over chains and draws)
    # Shape is (chains, draws, n_obs)
    mean_pred = pred_samples.mean(dim=("chain", "draw")).values
    lower_ci = pred_samples.quantile(0.025, dim=("chain", "draw")).values
    upper_ci = pred_samples.quantile(0.975, dim=("chain", "draw")).values
    
    # Add to DataFrame
    target_df['pred_mean'] = mean_pred
    target_df['pred_lower'] = lower_ci
    target_df['pred_upper'] = upper_ci
    
    print("Predictions generated.")
    return ppc, target_df # (arviz.InferenceData, pd.DataFrame)

def evaluate_model(df_with_preds, truth_col='disease_cases'):
    
    if truth_col not in df_with_preds.columns:
        print(f"Truth column '{truth_col}' not found. Cannot evaluate.")
        return {}
    
    y_true = df_with_preds[truth_col]
    y_pred = df_with_preds['pred_mean']
    
    # Drop NAs if any (e.g. if future data has no truth)
    valid_mask = ~y_true.isna()
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        return {"Error": "No valid ground truth data"}

    # Metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Coverage (if CI columns exist)
    coverage = None
    if 'pred_lower' in df_with_preds.columns and 'pred_upper' in df_with_preds.columns:
        lower = df_with_preds['pred_lower'][valid_mask]
        upper = df_with_preds['pred_upper'][valid_mask]
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "Coverage_95": coverage if coverage is not None else "N/A"
    }
    
    print(f"Evaluation Results: {metrics}")
    return metrics
