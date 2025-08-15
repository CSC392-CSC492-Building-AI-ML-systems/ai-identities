import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm


VECTORIZER_MAP = {
    'CountVectorizer': CountVectorizer,
    'TfidfVectorizer': TfidfVectorizer
}


def load_fitted_vectorizer(method_name: str, split_name: str = 'train'):
    """
    Load pre-fitted vectorizer from pickle if exists.
    """
    input_path = f"../data/processed/{method_name}/{split_name}/vectorizer.pkl"
    if os.path.exists(input_path):
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    return None


def process_word_freq(data: dict[str, pd.DataFrame], config: dict,
                      output_path: str, fitted_vectorizer=None,
                      drop_response: bool = True) -> tuple[dict[str, pd.DataFrame], object]:
    """
    Convert responses to word freq vectors (e.g., BoW or TF-IDF) and return processed dict and vectorizer.
    If output_path provided, also saves. If fitted_vectorizer provided, use it (skip fit).
    If drop_response is False, retain the 'response' column in the output DataFrames.
    """
    vectorizer_str = config['vectorizer']
    vectorizer_class = VECTORIZER_MAP.get(vectorizer_str)
    if not vectorizer_class:
        raise ValueError(f"Unknown vectorizer: {vectorizer_str}")

    # Default to unigrams; use config value if present
    ngram_range = tuple(config.get('ngram_range', (1, 1)))
    max_features = config.get('max_features', None)
    vectorizer = vectorizer_class(ngram_range=ngram_range, max_features=max_features)

    do_fit = fitted_vectorizer is None
    if not do_fit:
        vectorizer = fitted_vectorizer

    # Fit on all responses across models for consistent vocabulary (if do_fit)
    all_responses = pd.concat([df['response'] for df in data.values()])
    if do_fit:
        vectorizer.fit(all_responses)

    processed_data = {}
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    for llm_name, df in tqdm(data.items(), desc="Vectorizing LLM responses"):
        sparse_vectors = vectorizer.transform(df['response'])
        response_vectors = [sparse_vectors[i] for i in range(sparse_vectors.shape[0])]
        processed_df = df.copy()
        processed_df['response_vector'] = response_vectors  # list of csr_matrix
        if drop_response:
            processed_df = processed_df.drop(columns=['response'])
        processed_data[llm_name] = processed_df

        if output_path:
            # Save as pickle instead of JSON; JSON can't handle sparse matrix
            with open(os.path.join(output_path, f"{llm_name}.pkl"), 'wb') as f:
                pickle.dump(processed_df, f)

    # If we fitted and output_path provided, save the vectorizer
    if do_fit and output_path:
        with open(os.path.join(output_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

    return processed_data, vectorizer if do_fit else None


def process_embeddings(data: dict[str, pd.DataFrame], config: dict,
                       output_path: str) -> dict[str, pd.DataFrame]:
    """
    Convert responses to embedding vectors and return processed dict.
    If output_path provided, also saves.
    """
    embedding_model_name = config['model']
    embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)

    processed_data = {}
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    for llm_name, df in tqdm(data.items(), desc="Processing LLMs", unit="model", leave=True):
        vectors = embedding_model.encode(df['response'].tolist()).tolist()  # list of lists
        processed_df = df.copy()
        processed_df['response_vector'] = vectors
        processed_df = processed_df.drop(columns=['response'])
        processed_data[llm_name] = processed_df

        if output_path:
            with open(os.path.join(output_path, f"{llm_name}.pkl"), 'wb') as f:
                pickle.dump(processed_df, f)

    return processed_data


def process_and_save(split_data: dict[str, pd.DataFrame], clf_config: dict,
                     clf_method_name: str, split_name: str) -> dict[str, pd.DataFrame]:
    """
    Process and save to data/processed/{method_name}/{split_name}/.
    For word freq non-train splits, use train's fitted vectorizer if available.
    """
    output_path = f"../data/processed/{clf_method_name}/{split_name}/"
    if 'vectorizer' in clf_config:
        fitted_vectorizer = None
        if split_name != 'train':
            fitted_vectorizer = load_fitted_vectorizer(clf_method_name, 'train')
            if fitted_vectorizer is None:
                raise FileNotFoundError(f"Fitted vectorizer from train not found for {clf_method_name}")
        processed, _ = process_word_freq(split_data, clf_config, output_path, fitted_vectorizer)
        return processed
    else:
        return process_embeddings(split_data, clf_config, output_path)


def load_processed(clf_method_name: str, split_name: str) -> dict[str, pd.DataFrame]:
    """
    Load preprocessed data from data/processed/{method_name}/{split_name}/.
    Raises FileNotFoundError if not exists.
    """
    input_path = f"../data/processed/{clf_method_name}/{split_name}/"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Preprocessed data not found for {clf_method_name}/{split_name}")

    data = {}
    for file in os.listdir(input_path):
        if file.endswith('.pkl') and file != 'vectorizer.pkl' and file != 'library_averages.pkl':
            llm_name = file.replace('.pkl', '')
            with open(os.path.join(input_path, file), 'rb') as f:
                df = pickle.load(f)
            df['model'] = llm_name
            data[llm_name] = df

    if not data:
        # This can happen if the directory exists but is empty or only contains a vectorizer
        raise FileNotFoundError(f"No processed data files found for {clf_method_name}/{split_name}")

    return data
