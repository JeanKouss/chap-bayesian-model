"""
CHAP model: malaria-chap-project

A bayesian model implementation
"""

import joblib
import pandas as pd
from cyclopts import App
import arviz as az
from utils import BayesianModelUtils

DRAWS = 1000
TUNE = 1000

app = App()


@app.command()
def train(train_data: str, model: str):
    """Train the model on the provided data.

    Parameters
    ----------
    train_data
        Path to the training data CSV file.
    model
        Path where the trained model will be saved.
    """
    print("No training implemented")


@app.command()
def predict(model: str, historic_data: str, future_data: str, out_file: str):
    """Generate predictions using the trained model.

    Parameters
    ----------
    model
        Path to the trained model file.
    historic_data
        Path to historic data CSV file (unused in this simple model).
    future_data
        Path to future climate data CSV file.
    out_file
        Path where predictions will be saved.
    """
    df = pd.read_csv(historic_data)
    data_dict = BayesianModelUtils.load_and_preprocess_training_data(df, 'constants/districts.geojson', [
        "smc_number", "rainfall", "mean_temperature", "rel_humidity", "population", "area", "median_elevation"
    ])

    meta = {
        'gdf': data_dict['gdf'],
        'location_to_idx': data_dict['location_to_idx'],
        'covariates': data_dict['covariates'],
        'scaler': data_dict['scaler'],
        'n_locations': data_dict['n_locations'],
    }
    training_meta = {
        'training_date_to_idx': data_dict['training_date_to_idx'],
        'training_n_times': data_dict['training_n_times'],
        'n_training_obs': data_dict['n_training_obs'],
    }
    training_df = data_dict['df']
    # Compute adjancency matrix
    adjency_matrix, deg_matrix, gdf_reordered = BayesianModelUtils.compute_adjacency_matrix(meta['gdf'], meta['location_to_idx'])
    # Build and train the model
    max_test_times = 30
    py_mc_model = BayesianModelUtils.build_malaria_model(training_df, meta['covariates'], meta['n_locations'], training_meta['training_n_times'] + max_test_times, adjency_matrix)
    trace = BayesianModelUtils.train_model(py_mc_model, draws=DRAWS, tune=TUNE)

    future_df = pd.read_csv(future_data)
    future_df_prepared = BayesianModelUtils.prepare_prediction_data(future_df, meta['scaler'], meta['location_to_idx'], training_meta['training_date_to_idx'], meta['covariates'])

    future_ppc, future_prediction = BayesianModelUtils.predict_malaria(py_mc_model, meta['covariates'], trace, future_df_prepared)
    future_distribution_matrix = az.extract(future_ppc, group="predictions", var_names=["likelihood"]).values

    output_df = future_df_prepared[['location', 'time_period']]
    output_df['time_period'] = output_df['time_period'].dt.strftime('%Y-%m')
    for s in range(future_distribution_matrix.shape[1]):
        output_df[f'sample_{s}'] = future_distribution_matrix[:, s]

    output_df.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")


if __name__ == "__main__":
    app()
