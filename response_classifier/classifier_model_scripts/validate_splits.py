import pickle
import pandas as pd
import yaml


def validate_splits(train: dict[str, pd.DataFrame], test: dict[str, pd.DataFrame],
                    held_out: dict[str, pd.DataFrame], bins: dict,
                    split_ratio: int) -> bool:
    """
    Validate splits: check sizes, bin presence, and ratios.
    """
    all_llms = set(train.keys()) | set(test.keys()) | set(held_out.keys())
    total_number_of_prompts = 20
    # For each LLM in our dataset
    for model in all_llms:
        if model in held_out:
            print(f"{model} is held-out: {len(held_out[model])} rows")
            continue

        train_df = train.get(model, pd.DataFrame())
        test_df = test.get(model, pd.DataFrame())
        # For each temperature bin
        for bin_name, bin_info in bins.items():
            train_bin = train_df[train_df['temp_bin'] == bin_name]
            test_bin = test_df[test_df['temp_bin'] == bin_name]
            expected_train = total_number_of_prompts * int(bin_info['num_points'] * split_ratio / (split_ratio + 1))
            expected_test = total_number_of_prompts * int(bin_info['num_points'] / (split_ratio + 1))
            if len(train_bin) != expected_train or len(test_bin) != expected_test:
                print(f"Size mismatch for {model}, {bin_name}")
                print(f"Expected train: {expected_train}, actual train: {len(train_bin)}")
                print(f"Expected test: {expected_test}, actual test: {len(test_bin)}")
                return False

            if len(train_bin) == 0 or len(test_bin) == 0:
                print(f"Missing bin {bin_name} for {model}")
                return False

    print("All splits validated successfully.")

    return True


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/data_config.yaml'))
    with open('data/splits/train.pkl', 'rb') as f:
        train = pickle.load(f)

    with open('data/splits/test.pkl', 'rb') as f:
        test = pickle.load(f)

    with open('data/splits/held_out.pkl', 'rb') as f:
        held_out = pickle.load(f)

    validate_splits(train, test, held_out, config['temp_bins'], config['split_ratio'])