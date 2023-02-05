from typing import List
import pandas as pd
import numpy as np
import os
import sys
import shutil
import json
import math
import datetime
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import norm


DATA_PATH = os.path.join(os.getcwd(), 'model_data')
OUTPUT_PATH = os.path.join(os.getcwd(), 'output')
# DO NOT MODIFY
GAME_IDENTIFIERS = [
    'GAME_ID',
    'GAME_DATE'
]
TEAM_IDENTIFIER = 'TEAM_ID'
BEST_MAPE = 0.0842
# MODIFY
YEAR_FILTER = 2000     # Only filter games > given year, use None for no filter
GAME_LAG = sorted([1,2, 3, 4, 5, 6, 7, 8, 9, 10])
PREDICTORS = [
    'FGA',
    'FG_PCT',
    'FG3A',
    'FG3_PCT',
    'FTA',
    'FT_PCT',
    'OREB',
    'DREB',
    'AST',
    'STL',
    'BLK',
    'TOV',
    'PF',
    'LAG_WINS',
    'LAG_LOSSES'
]
TARGET = 'PTS'
TRAINING_FRAC = 0.95
SPLIT_METHOD = 'RANDOM'      # ['TIME', 'RANDOM']
SUPPORTED_MODELS = {
    'Gradient Boosting Regressor': HistGradientBoostingRegressor(),        # In the future would be good to do some hyperparameter tuning using RayTune
    'K Nearest Neighbors': KNeighborsRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Linear Regressor': LinearRegression()
}


def extract_data(game_lag) -> pd.DataFrame:
    try:
        data_df = pd.read_parquet(os.path.join(DATA_PATH, f'game_lag_{game_lag}.parquet'))
    except FileNotFoundError as e:
        print(e)
        print(f'game_lag {game_lag} data not found, see README rebuilding model_data with required game_lag')
        exit(1)
    data_df['GAME_DATE'] = pd.to_datetime(data_df['GAME_DATE'])
    if YEAR_FILTER is not None:
        data_df = data_df[data_df['GAME_DATE'] > datetime.datetime(YEAR_FILTER, 1, 1)]
    for k in PREDICTORS + [TARGET]:
        data_df[k] = pd.to_numeric(data_df[k])
    data_df.sort_values(by='GAME_DATE', inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    return data_df


def split_random(
    seed: int,
    data_df: pd.DataFrame,
):
    data_rows = data_df.shape[0]
    training_df = data_df.sample(n=round(data_rows*TRAINING_FRAC), random_state=seed)
    testing_df = data_df[~data_df.index.isin(training_df.index)]
    training_df.reset_index(drop=True, inplace=True)
    testing_df.reset_index(drop=True, inplace=True)
    return (
        training_df[PREDICTORS],
        training_df[TARGET],
        testing_df[PREDICTORS],
        testing_df[TARGET]
    )


def split_data_by_time(
    data_df: pd.DataFrame,
):
    data_rows = data_df.shape[0]
    training_df = data_df.head(round(data_rows*TRAINING_FRAC))
    testing_df = data_df[~data_df.index.isin(training_df.index)]
    testing_df.reset_index(drop=True, inplace=True)
    return (
        training_df[PREDICTORS],
        training_df[TARGET],
        testing_df[PREDICTORS],
        testing_df[TARGET]
    )


class Model:
    def __init__(
        self,
        name,
        reg,
        in_sample_mape,
        out_sample_mape
    ):
        self.name = name
        self.reg = reg
        self.in_sample_mape = in_sample_mape
        self.out_sample_mape = out_sample_mape

def train_models(
    training_predictors_df: pd.DataFrame,
    training_target_df: pd.Series,
    testing_predictors_df: pd.DataFrame,
    testing_target_df: pd.Series
) -> List:
    models = []
    for name, technique in SUPPORTED_MODELS.items():
        print(name)
        reg = technique.fit(training_predictors_df, training_target_df)
        in_sample_mape = (1 / training_predictors_df.shape[0]) * (
            np.sum(
                abs((training_target_df - reg.predict(training_predictors_df)) / reg.predict(training_predictors_df))
            )
        )
        out_sample_mape = (1 / testing_predictors_df.shape[0]) * (
            np.sum(
                abs((testing_target_df - reg.predict(testing_predictors_df)) / reg.predict(testing_predictors_df))
            )
        )
        models.append(
            Model(
                name=name, 
                reg=reg,
                in_sample_mape=in_sample_mape,
                out_sample_mape=out_sample_mape
            )
        )
    return models


def plot_error_distribution(
    predictor_errors: dict,
    out_path: str
):
    cols = 4
    rows = math.ceil(len(predictor_errors) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(7.5*cols, 5*rows))
    fig.suptitle(f'Basketball Score Prediction Error Distributions', fontsize=14, fontweight='bold')

    i = 0
    for idx, (name, error_df) in enumerate(predictor_errors.items()):
        j = idx % cols
        if idx != 0 and j == 0:
            i += 1
        tmp_axs = axs[i, j] if len(predictor_errors) > cols else axs[j]
        _, bins, _ = tmp_axs.hist(error_df, bins=100, density=True, color='red', alpha=0.7)
        mu, sigma = norm.fit(error_df)
        best_fit = norm.pdf(bins, mu, sigma)
        tmp_axs.plot(bins, best_fit, linestyle='-', color='black', alpha=0.7)
        mean = error_df.mean()
        std = error_df.std()
        mean_line = tmp_axs.axvline(x=mean, linestyle='--', color='blue', alpha=0.5)
        std_line = tmp_axs.axvline(x=(mean+std), linestyle='--', color='green', alpha=0.25)
        tmp_axs.axvline(x=(mean-std), linestyle='--', color='green', alpha=0.25)
        tmp_axs.set_title(f'{name} - Error Distribution', fontsize=12, fontweight='bold')
        tmp_axs.set_ylabel('Density', fontsize=10)
        tmp_axs.set_xlabel('Predicted Points Error', fontsize=10)
        tmp_axs.legend([mean_line, std_line], [f"Mean ({round(mean, 2)})", f"STD ({round(std, 2)})"])

    plt.subplots_adjust(wspace=0.40, hspace=0.40)
    plt.savefig(fname=out_path)


def plot_mape(
    mape_df: pd.DataFrame,
    out_path: str
):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Basketball Predictor MAPE Comparison', fontsize=12, fontweight='bold')
    mape_df.plot(kind='bar', ax=axs[0])
    axs[0].axhline(y=BEST_MAPE, linestyle='--', color='black', alpha=0.5)
    axs[0].set_ylabel('MAPE', fontsize=10)
    axs[0].tick_params(rotation=0)
    fig.text(0.12, 0.22, f'Black dashed line represents best MAPE discussed in article {BEST_MAPE}', fontsize=8)
    axs[1].table(
        cellText=mape_df.T.values,
        colLabels=mape_df.T.columns,
        rowLabels=mape_df.T.index,
        colWidths=np.full(len(mape_df.T.columns), 0.1),
        colColours=np.full(len(mape_df.T.columns), 'lavender'),
        rowColours=np.full(len(mape_df.T.index), 'linen')
    )
    axs[1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fname=out_path)


if __name__ == '__main__':
    seed = 0
    try:
        seed += int(sys.argv[1])
    except (TypeError, ValueError) as e:
        print(f'Please pass seed as number for example \'python {sys.argv[0]} 12345678\'')
        exit(1)
    except IndexError:
        pass
    
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)
    
    lag_experiments = []
    mape = []

    print('Starting...')
    for game_lag in GAME_LAG:
        print(f'\nBuilding models for game_lag {game_lag}')
        data_df = extract_data(game_lag)
        if SPLIT_METHOD == 'TIME':
            training_predictors_df, training_target_df, testing_predictors_df, testing_target_df = split_data_by_time(data_df)
        else:
            SPLIT_METHOD = 'RANDOM'
            training_predictors_df, training_target_df, testing_predictors_df, testing_target_df = split_random(seed, data_df)

        models = train_models(
            training_predictors_df, 
            training_target_df,
            testing_predictors_df,
            testing_target_df
        )
        model_metrics = []
        predictor_errors = {}
        lag_mape = []

        for model in models:
            error_df = pd.Series(testing_target_df - pd.Series(model.reg.predict(testing_predictors_df)))
            predictor_errors[model.name] = error_df
            lag_mape.append(round(model.out_sample_mape, 3))

            model_metrics.append({
                'name': model.name,
                'in_sample_mape': model.in_sample_mape,
                'out_sample_mape': model.out_sample_mape,
                'avg score prediction error': error_df.mean(),
                'std score prediction error': error_df.std()
            })
        
        plot_error_distribution(predictor_errors, os.path.join(OUTPUT_PATH, f'error_distribution_game_lag_{game_lag}.png'))
        mape.append(lag_mape)
        lag_experiments.append({
            'game_lag': game_lag,
            'dataset_count': {
                'total': training_predictors_df.shape[0] + testing_predictors_df.shape[0],
                'training': training_predictors_df.shape[0],
                'testing': testing_predictors_df.shape[0]
            },
            'models': model_metrics
        })
    
    metrics = {
        'model_name': 'Basketball Score Predictor',
        'description': 'https://www.mdpi.com/1099-4300/23/4/477',
        'year_filter': YEAR_FILTER,
        'training_frac': TRAINING_FRAC,
        'split_method': SPLIT_METHOD,
        'seed': seed,
        'lag_experiments': lag_experiments
    }
    mape_df = pd.DataFrame.from_records(mape, index=[f'Game Lag {l}' for l in GAME_LAG], columns=SUPPORTED_MODELS.keys())
    plot_mape(mape_df, os.path.join(OUTPUT_PATH, 'mape_comparison.png'))
    with open(os.path.join(OUTPUT_PATH, 'metrics.json'), 'w') as metrics_json:
        json.dump(metrics, metrics_json)
    print(f'Output {OUTPUT_PATH}')
