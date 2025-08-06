import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import yaml
from data_loader import load_raw_data


def bin_temperatures(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    """
    Add 'temp_bin' column to DataFrame based on temp bins.

    :param df: A Pandas DataFrame that contains response data for a specific LLM.
    :param bins: A dictionary defining the temperature bin ranges. For example,
        {'low': {'min': 0.0, 'max': 0.4, 'num_points': 12},
        'medium': {'min': 0.4, 'max': 0.8, 'num_points': 24},
        'high': {'min': 0.8, 'max': 1.2, 'num_points': 48}}.
    :return: A modified DataFrame with an additional 'temp_bin' column.
    """
    conditions = [
        (df['temperature'] >= bins['low']['min']) & (df['temperature'] < bins['low']['max']),
        (df['temperature'] >= bins['medium']['min']) & (df['temperature'] < bins['medium']['max']),
        (df['temperature'] >= bins['high']['min']) & (df['temperature'] < bins['high']['max'])
    ]
    choices = ['low', 'medium', 'high']
    df['temp_bin'] = np.select(conditions, choices, default='unknown')

    return df


def split_data_per_bin(data: dict[str, pd.DataFrame], bins: dict, split_ratio: int,
                       seed: int) -> tuple[dict, dict]:
    """
    Split dataset into train and test sets, using `split_ratio`:1 ratio, within each temp
    bin for each (model, prompt) pair. Returns dicts of train and test set DataFrames.
    """
    train_data, test_data = {}, {}
    test_size = 1 / (split_ratio + 1)  # e.g., 1/6 for 5:1
    for model, df in data.items():
        df = bin_temperatures(df, bins)
        curr_model_train_parts, curr_model_test_parts = [], []
        for (prompt, bin_name), group in df.groupby(['prompt', 'temp_bin']):
            expected_num_points = bins[bin_name]['num_points']
            actual_num_points = len(group)
            if actual_num_points != expected_num_points:
                raise ValueError(
                    f"Mismatch in data points for {model}, {prompt}, {bin_name}. "
                    f"Expected {expected_num_points} data points, but got {actual_num_points}.")

            train_group, test_group = train_test_split(group, test_size=test_size,
                                                       random_state=seed)
            curr_model_train_parts.append(train_group)
            curr_model_test_parts.append(test_group)

        train_data[model] = pd.concat(curr_model_train_parts)
        test_data[model] = pd.concat(curr_model_test_parts)

    return train_data, test_data


def hold_out_models(data: dict[str, pd.DataFrame], config: dict) -> tuple[dict, dict]:
    """
    Hold out models: Use list from config if present, else random fraction. Returns non_held_out, held_out dicts.
    """
    if 'held_out_models' in config and config['held_out_models']:
        held_out_models = config['held_out_models']
    else:
        np.random.seed(config['random_seed'])
        models = list(data.keys())
        num_hold_out = int(len(models) * config['hold_out_fraction'])
        held_out_models = np.random.choice(models, num_hold_out, replace=False)

    held_out_set = {m: data[m] for m in held_out_models if m in data}
    non_held_out_set = {m: data[m] for m in data if m not in held_out_set}

    return non_held_out_set, held_out_set


def save_splits(train: dict, test: dict, held_out: dict, output_path: str) -> None:
    """
    Save splits as pickled dicts.
    """
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(output_path, 'test.pkl'), 'wb') as f:
        pickle.dump(test, f)
    with open(os.path.join(output_path, 'held_out.pkl'), 'wb') as f:
        pickle.dump(held_out, f)


# Example usage (in main_experiment.py)
if __name__ == '__main__':
    config = yaml.safe_load(open('../configs/data_config.yaml'))
    raw = load_raw_data(config['raw_data_path'])  # From data_loader
    non_held_out, held_out = hold_out_models(raw, config)
    train, test = split_data_per_bin(non_held_out, config['temp_bins'],
                                     config['split_ratio'], config['random_seed'])
    save_splits(train, test, held_out, '../data/splits/')