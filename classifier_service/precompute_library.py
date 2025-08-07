import os
import pickle
import sys
# Add the project root directory to the Python path to allow imports from response_classifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from response_classifier.classifier_model_scripts.classifier_model import compute_library_averages


BEST_USER_PROMPT = ("Invent a new taste unknown to humans. Describe (in less than 800 "
                    "words) how it feels, what foods or cuisines feature it, and how it "
                    "could transform food culture and impact health. Speak like a "
                    "professor and only use vocabularies, wordings, etc professors use.")

# Path to the training data split, relative to the project root.
TRAIN_DATA_PATH = os.path.join('response_classifier', 'data', 'splits', 'train.pkl')

# Paths are relative to the project root, where the script should be executed from.
INFERENCE_DIR = 'inference'
ASSETS_DIR = os.path.join(INFERENCE_DIR, 'assets')
VECTORIZER_PATH = os.path.join(ASSETS_DIR, 'vectorizer.pkl')
LIBRARY_AVERAGES_PATH = os.path.join(ASSETS_DIR, 'library_averages.pkl')


def precompute_and_save_library_averages():
    """
    Loads the training data, computes average vectors for the best prompt,
    and saves them to a file for fast inference.
    """
    print("Starting library pre-computation...")

    # Load data and vectorizer
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Error: Training data not found at {TRAIN_DATA_PATH}")
        print(
            "Please ensure you have run the data splitting script and are running this from the project root.")
        sys.exit(1)
    if not os.path.exists(VECTORIZER_PATH):
        print(
            f"Error: Vectorizer not found at {VECTORIZER_PATH}. Make sure to copy it to the assets folder.")
        sys.exit(1)

    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_data = pickle.load(f)
    print(f"Loaded training data for {len(train_data)} models.")

    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Loaded fitted TF-IDF vectorizer.")

    # Filter and process data for the best prompt
    library_for_prompt = {}
    for model, df in train_data.items():
        prompt_df = df[df['prompt'] == BEST_USER_PROMPT].copy()
        if not prompt_df.empty:
            # Vectorize responses
            vectors = vectorizer.transform(prompt_df['response'])
            prompt_df['response_vector'] = [vectors[i] for i in range(vectors.shape[0])]
            library_for_prompt[model] = prompt_df

    if not library_for_prompt:
        print(f"Error: The specified prompt '{BEST_USER_PROMPT}' was not found in the training data.")
        sys.exit(1)

    print(f"Filtered and processed data for the target prompt for {len(library_for_prompt)} models.")

    # Compute averages
    library_averages = compute_library_averages(library_for_prompt)

    # Store averages with a simpler key for this specific prompt: (model, bin_name)
    prompt_specific_averages = {}
    for (model, prompt, bin_name), avg_vec in library_averages.items():
        if prompt == BEST_USER_PROMPT:
            prompt_specific_averages[(model, bin_name)] = avg_vec

    print(f"Computed {len(prompt_specific_averages)} average vectors for the library.")

    # Save the result
    os.makedirs(ASSETS_DIR, exist_ok=True)
    with open(LIBRARY_AVERAGES_PATH, 'wb') as f:
        pickle.dump(prompt_specific_averages, f)

    print(f"Successfully saved pre-computed library averages to {LIBRARY_AVERAGES_PATH}")


if __name__ == '__main__':
    precompute_and_save_library_averages()
