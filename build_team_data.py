import pandas as pd
import os
import sys


INPUT_PATH = os.path.join(os.getcwd(), 'extracted_raw_data')
OUTPUT_PATH = os.path.join(os.getcwd(), 'model_data')
# DO NOT MODIFY
GAME_IDENTIFIERS = [
    'GAME_ID',
    'GAME_DATE'
]
TEAM_IDENTIFIER = 'TEAM_ID'
# MODIFY
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
    'PF'
]
APPENDED_PREDICTORS = [
    'SEASON_WINS',
    'SEASON_LOSSES'
]
TARGET = [
    'PTS',
    'WON'
]


def extract_data() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(INPUT_PATH, 'game.parquet'))


def extract_games(data_df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for s in ['HOME', 'AWAY']:
        # Generate custom columns
        data_df[f'WON_{s}'] = data_df.apply(lambda r: r[f'PTS_{s}'] > r[f'PTS_{"HOME" if s == "AWAY" else "AWAY"}'], axis=1)
        data_df[[f'SEASON_WINS_{s}', f'SEASON_LOSSES_{s}']] = data_df[f'TEAM_WINS_LOSSES_{s}'].str.split('-', expand=True)
        data_df[f'SEASON_WINS_{s}'] = data_df[f'SEASON_WINS_{s}'].replace('', 0)
        data_df[f'SEASON_LOSSES_{s}'] = data_df[f'SEASON_LOSSES_{s}'].replace('', 0)
        
        # Isolate columns
        rename_dict = {f'{k}_{s}':k for k in PREDICTORS + APPENDED_PREDICTORS + [TEAM_IDENTIFIER] + TARGET}
        isolated_columns = list(rename_dict.keys()) + GAME_IDENTIFIERS
        s_df = data_df[isolated_columns]
        s_df = s_df.rename(columns=rename_dict)
        dfs.append(s_df)
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    merged_df['GAME_DATE'] = pd.to_datetime(merged_df['GAME_DATE'])
    for k in PREDICTORS + TARGET:
        merged_df[k] = pd.to_numeric(merged_df[k])
    merged_df.sort_values(by='GAME_DATE', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


def build_lagged_stats(
    game_lag: int,
    games_df: pd.DataFrame,
    game: pd.Series
) -> dict:
    if game.name % 1000 == 0:
        print(f"Processed INDEX={game.name} GAME_DATE={game['GAME_DATE']} GAMES_DATAFRAME_SIZE={games_df.shape[0]}")

    filtered_games_df = games_df[games_df['TEAM_ID'] == game['TEAM_ID']].tail(game_lag)
    if filtered_games_df.isnull().values.any() or game.isnull().values.any() or filtered_games_df.shape[0] < game_lag:
        return None
    else:
        # PREDICTORS
        data = {k:filtered_games_df[k].mean() for k in PREDICTORS}

        # Generate lagged form
        data['LAG_WINS'] = len(filtered_games_df[filtered_games_df['WON'] == True])
        data['LAG_LOSSES'] = len(filtered_games_df[filtered_games_df['WON'] == False])

        # Generate season form
        data['SEASON_WINS'] = int(game['SEASON_WINS'])
        data['SEASON_LOSSES'] = int(game['SEASON_LOSSES'])

        for k in TARGET + [TEAM_IDENTIFIER] + GAME_IDENTIFIERS:
            data[k] = game[k]
        return data


if __name__ == '__main__':
    # Get arg
    try:
        game_lag = int(sys.argv[1])
    except (IndexError, ValueError) as e:
        print('Please pass lag argument')
        exit(1)

    if game_lag <= 0:
        print('Lag must be > 0')
        exit(1)
    print(f'game lag {game_lag}')

    # Get base game data
    data_df = extract_data()
    games_df = extract_games(data_df)
    print(f'{games_df.shape[0]} rows to process')

    # Collect game stats
    iter_games_df = games_df.copy()
    lagged_games_stats = []
    for i in reversed(games_df.index):
        row = games_df.loc[i]
        iter_games_df = iter_games_df[iter_games_df['GAME_DATE'] < row['GAME_DATE']]
        res = build_lagged_stats(game_lag, iter_games_df, row)
        if res is not None:
            lagged_games_stats.append(res)
    lagged_games_stats_df = pd.DataFrame.from_records(lagged_games_stats, columns=lagged_games_stats[0].keys())

    # Save output
    SAVE_PATH = os.path.join(OUTPUT_PATH, f'game_lag_{game_lag}.parquet') 
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
    lagged_games_stats_df.to_parquet(SAVE_PATH)
    print(f'Saved {lagged_games_stats_df.shape[0]} rows')
