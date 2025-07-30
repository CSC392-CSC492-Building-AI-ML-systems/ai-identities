import yaml
import argparse
from data_loader import load_raw_data
from data_processor import process_and_save
from data_splitter import split_data_per_bin, hold_out_models, save_splits
from cross_validation import perform_5fold_cv_for_method
from evaluator import evaluate_final_model
import pickle
from validate_splits import validate_splits


def parse_arguments():
    parser = argparse.ArgumentParser(description="Main experiment script for LLM response classification.")
    parser.add_argument('--action', type=str, required=True,
                        choices=['split_data', 'validate_splits', 'cross_validation', 'evaluate', 'process'],
                        help="Action to perform: 'split_data', 'validate_splits', 'cross_validation' (requires --method), 'evaluate' (requires --method), 'process'.")
    parser.add_argument('--method', type=str, default=None,
                        help="Classification method name (e.g., 'nomic_cosine') for 'cross_validation' or 'evaluate' actions.")
    parser.add_argument('--eval_both', action='store_true',
                        help="If set, evaluate CV with both cosine and euclidean metrics.")
    return parser.parse_args()


def main(args):
    data_cfg = yaml.safe_load(open('../configs/data_config.yaml'))
    methods_cfg = yaml.safe_load(open('../configs/classification_methods_config.yaml'))
    raw, _, _, _ = load_raw_data(data_cfg['raw_data_path'])

    if args.action == 'split_data':
        non_held_out, held_out = hold_out_models(raw, data_cfg)
        train, test = split_data_per_bin(non_held_out, data_cfg['temp_bins'],
                                         data_cfg['split_ratio'],
                                         data_cfg['random_seed'])
        save_splits(train, test, held_out, '../data/splits/')

    if args.action == 'validate_splits':
        with open('../data/splits/train.pkl', 'rb') as f:
            train = pickle.load(f)
        with open('../data/splits/test.pkl', 'rb') as f:
            test = pickle.load(f)
        with open('../data/splits/held_out.pkl', 'rb') as f:
            held_out = pickle.load(f)
        validate_splits(train, test, held_out, data_cfg['temp_bins'], data_cfg['split_ratio'])

    if args.action == 'cross_validation':
        if not args.method:
            raise ValueError("For 'cross_validation' action, --method is required (e.g., nomic_cosine)")

        classification_method = next((m for m in methods_cfg['embeddings'] + methods_cfg['word_freq']
                                      if m['name'] == args.method), None)
        if not classification_method:
            raise ValueError(f"Method '{args.method}' not found in config")
        with open('../data/splits/train.pkl', 'rb') as f:
            train = pickle.load(f)
        perform_5fold_cv_for_method(train, classification_method, eval_both_metrics=args.eval_both)

    if args.action == 'evaluate':
        if not args.method:
            raise ValueError("For 'evaluate' action, --method is required (e.g., nomic_cosine)")

        classification_method = next((m for m in methods_cfg['embeddings'] + methods_cfg['word_freq']
                                      if m['name'] == args.method), None)
        if not classification_method:
            raise ValueError(f"Method '{args.method}' not found in config")

        # Load splits
        with open('../data/splits/train.pkl', 'rb') as f:
            train = pickle.load(f)
        with open('../data/splits/test.pkl', 'rb') as f:
            test = pickle.load(f)
        with open('../data/splits/held_out.pkl', 'rb') as f:
            held_out = pickle.load(f)
        evaluate_final_model(classification_method, train, test, held_out)

    if args.action == 'process':
        with open('../data/splits/train.pkl', 'rb') as f:
            train = pickle.load(f)

        # Process and save for all methods
        all_methods = methods_cfg['word_freq'] + methods_cfg['embeddings']
        for method_cfg in all_methods:
            process_and_save(train, method_cfg, method_cfg['name'], 'train')
        print("All methods processed and saved for the train set.")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)