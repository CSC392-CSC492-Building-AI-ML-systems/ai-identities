import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

# Constants for training
TRAINING_WORDS_LIST = ['life-filled', 'home.', 'wondrous', 'immense', 'ever-changing.', 'massive', 
                       'enigmatic', 'complex.', 'finite.\n\n\n\n', 'lively', 
                       "here'sadescriptionofearthusingtenadjectives", 'me', 'dynamic', 'beautiful', 
                       'ecosystems', 'interconnected.', 'finite.', 'big', '10', 'nurturing', 'then', 
                       '"diverse"', 'are', 'verdant', 'diverse', 'life-giving', 'lush', 'here', '8.', 
                       'ten', 'and', 'powerful', 'precious.', "it's", 'mysterious', 'temperate', 
                       'evolving', 'resilient', 'think', 'intricate', 'by', 'breathtaking.', 'varied', 
                       'commas:', 'evolving.', 'describe', 'essential.', 'arid', 'i', 'separated', 
                       'adjectives', 'orbiting', 'a', 'inhabited', '6.', 'revolving', 'nurturing.', 
                       'need', 'swirling', 'home', 'life-supporting', '10.', 'bountiful', 'because', 
                       'fertile', 'resilient.\n\n\n\n', 'precious.\n\n\n\n', 'should', 'old', 'hmm', 
                       'watery', 'thriving', 'magnificent', 'life-sustaining', 'adjectives:', 'exactly', 
                       'spherical', 'okay', 'earth', 'resilient.', 'the', 'only', 'beautiful.', 
                       'turbulent', 'start', 'terrestrial', 'teeming.', 'its', 'life-giving.', 'dense', 
                       'teeming', 'resourceful', 'ancient', 'round', '1.', 'using', 'about', 'rocky', 
                       'comma.', 'volatile', 'brainstorming', 'habitable.', 'to', 'in', 'stunning', 
                       'fascinating', 'abundant', 'habitable', 'aquatic', 'hospitable', 'volcanic', 
                       'let', 'awe-inspiring', 'changing', '2.', 'landscapes', 'awe-inspiring.', 'of', 
                       'magnetic', 'breathtaking', 'alive.', 'is', 'layered', 'planet', 'beautiful.\n\n\n\n', 
                       'majestic.', 'alive', 'mountainous', 'active', 'enigmatic.', 'our', 
                       'irreplaceable.', 'fragile', 'blue', 'mysterious.', 'each', 'huge', 
                       'interconnected', 'separatedbycommas:\n\nblue', 'rugged', 'barren', 'so', 
                       'atmospheric', 'mind', 'vital', 'finite', 'fragile.', 'inhabited.', 'first', 
                       'wants', 'description', 'ever-changing', 'chaotic', 'blue.', 'vast', '', 
                       'habitable.\n\n\n\n', 'precious', 'rotating', 'warm', 'large', 'spinning', 
                       'expansive', '7.', 'solid', 'vibrant', 'green', 'wet', 'extraordinary.', 
                       'user', 'complex', 'wondrous.', 'majestic', 'comes', 'unique', 'unique.', 
                       'life-sustaining.', 'living']

# Create a word-to-index mapping for fast lookups and preserving order
TRAINING_WORDS_DICT = {word: idx for idx, word in enumerate(TRAINING_WORDS_LIST)}
TRAINING_WORDS_SET = set(TRAINING_WORDS_LIST)

# List of models in training data with their order
LIST_OF_MODELS = ["chatgpt-4o-latest", "DeepSeek-R1-Distill-Llama-70B", "DeepSeek-R1-Turbo", 
                 "DeepSeek-R1", "DeepSeek-V3", "gemini-1.5-flash", "gemini-2.0-flash-001", 
                 "gemma-2-27b-it", "gemma-3-27b-it", "gpt-3.5-turbo", "gpt-4.5-preview", 
                 "gpt-4o-mini", "gpt-4o", "Hermes-3-Llama-3.1-405B", "L3.1-70B-Euryale-v2.2", 
                 "L3.3-70B-Euryale-v2.3", "Llama-3.1-Nemotron-70B-Instruct", 
                 "Llama-3.2-90B-Vision-Instruct", "Llama-3.3-70B-Instruct-Turbo", 
                 "Meta-Llama-3.1-70B-Instruct-Turbo", "Mistral-Nemo-Instruct-2407", 
                 "Mixtral-8x7B-Instruct-v0.1", "MythoMax-L2-13b", "o1-mini", 
                 "Phi-4-multimodal-instruct", "phi-4", "Qwen2.5-7B-Instruct", 
                 "Sky-T1-32B-Preview", "WizardLM-2-8x22B"]

MODEL_TO_INDEX = {model: idx for idx, model in enumerate(LIST_OF_MODELS)}
MODEL_SET = set(LIST_OF_MODELS)

def load_heatmap_data(file_path):
    """Load heatmap data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            heatmap_data = json.load(f)
        return heatmap_data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

def prepare_features(heatmap_data, for_training=True):
    """
    Prepare feature matrix from heatmap data
    - Creates a DataFrame with models as rows and words as columns
    - Each cell contains the normalized frequency of that word in that model
    """
    # Extract frequencies table
    normalized_frequencies = pd.DataFrame(heatmap_data['normalized_frequencies'])
    
    # Ensure proper normalization within each model (row)
    normalized_data = normalized_frequencies.div(normalized_frequencies.sum(axis=1), axis=0)
    
    if for_training:
        # For training data, just verify we have the expected columns
        missing_words = [word for word in TRAINING_WORDS_LIST if word not in normalized_data.columns]
        if missing_words:
            print(f"Warning: {len(missing_words)} words from TRAINING_WORDS_LIST not found in training data")
            print(f"First few missing: {missing_words[:5]}")
            
            # Add missing columns with zeros
            for word in missing_words:
                normalized_data[word] = 0.0
    
    # Display info about the normalized data
    print(f"Feature matrix shape: {normalized_data.shape} (models × words)")
    
    return normalized_data

def align_validation_data(validation_data):
    """
    Align validation data with training data:
    1. Remove words in validation not in training
    2. Add words from training missing in validation with 0 values
    3. Ensure columns are in the same order as training data
    """
    # Step 1: Keep only columns that exist in the training data
    valid_columns = [col for col in validation_data.columns if col in TRAINING_WORDS_SET]
    columns_to_remove = [col for col in validation_data.columns if col not in TRAINING_WORDS_SET]
    print(f"Removing {len(columns_to_remove)} words from validation data that don't exist in training")
    if columns_to_remove:
        print(f"Examples of removed words: {columns_to_remove[:5]}")
    
    # Step 2: Create a new DataFrame with all training words
    aligned_data = pd.DataFrame(index=validation_data.index, columns=TRAINING_WORDS_LIST)
    aligned_data.fillna(0.0, inplace=True)
    
    # Step 3: Copy values from validation data for words that exist in both
    for col in valid_columns:
        aligned_data[col] = validation_data[col]
    
    # Step 4: Sort columns to match training word order
    # This step is already done by creating the DataFrame with TRAINING_WORDS_LIST
    
    print(f"Validation data aligned:")
    print(f"- Original shape: {validation_data.shape}")
    print(f"- Aligned shape: {aligned_data.shape}")
    print(f"- Words removed: {len(columns_to_remove)}")
    print(f"- Words added: {len(TRAINING_WORDS_LIST) - len(valid_columns)}")
    
    return aligned_data

def align_model_order(data_df):
    """
    Align models in the validation data to match the order in training data
    Add missing models with zero vectors
    """
    # Start with all models in correct order
    aligned_df = pd.DataFrame(index=LIST_OF_MODELS, columns=data_df.columns)
    aligned_df.fillna(0.0, inplace=True)
    
    # Copy data for models that exist in both
    common_models = [model for model in data_df.index if model in MODEL_SET]
    for model in common_models:
        aligned_df.loc[model] = data_df.loc[model]
    
    print(f"Model alignment complete:")
    print(f"- Original models: {len(data_df.index)}")
    print(f"- Common models: {len(common_models)}")
    print(f"- Final aligned models: {len(aligned_df.index)}")
    
    return aligned_df

def train_and_validate(train_file_path, validation_file_path):
    """Train on one heatmap file and validate on another"""
    # Load training data
    train_heatmap = load_heatmap_data(train_file_path)
    if train_heatmap is None:
        return None, None, None, None
    
    # Prepare training features
    train_data = prepare_features(train_heatmap, for_training=True)
    
    # Ensure training data has exactly the right columns in the right order
    train_data = train_data[TRAINING_WORDS_LIST]
    
    # Load validation data
    validation_heatmap = load_heatmap_data(validation_file_path)
    if validation_heatmap is None:
        return None, None, None, None
    
    # Prepare validation features
    validation_data = prepare_features(validation_heatmap, for_training=False)
    
    # Align validation data words with training data
    aligned_validation = align_validation_data(validation_data)
    
    # Align model order to match training data
    aligned_validation = align_model_order(aligned_validation)
    
    # Extract numpy arrays for training and validation
    X_train = train_data.values
    y_train = train_data.index.tolist()
    
    X_val = aligned_validation.values
    y_val = aligned_validation.index.tolist()
    
    # Verify dimensions
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Train the classifiers
    print("Training classifiers...")
    clf = RandomForestClassifier(max_depth=15, n_estimators=400, random_state=42)
    clf2 = DecisionTreeClassifier(max_depth=15, random_state=42)
    clf3 = LogisticRegression(random_state=42)
    clf4 = SVC(random_state=42)
    clf5 = SVC(kernel='linear', random_state=42)
    clf6 = SVC(kernel='poly', random_state=42)
    
    # Neural Network classifier
    clf7 = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
        activation='relu',             # ReLU activation function
        solver='adam',                 # Adam optimizer
        alpha=0.0001,                  # L2 penalty parameter
        batch_size='auto',             # Automatic batch size
        learning_rate='adaptive',      # Adaptive learning rate
        max_iter=1000,                 # Maximum number of iterations
        early_stopping=False,           # Use early stopping
        validation_fraction=0.1,       # Fraction of training data for validation
        n_iter_no_change=10,           # Number of iterations with no improvement
        random_state=42                # Random state for reproducibility
    )
    
    # Fit all classifiers
    clf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)
    clf4.fit(X_train, y_train)
    clf5.fit(X_train, y_train)
    clf6.fit(X_train, y_train)
    clf7.fit(X_train, y_train)
    print("Training complete!")
    
    # Find which validation models actually have data (not all zeros)
    has_data = np.sum(X_val, axis=1) > 0
    active_models = [model for model, active in zip(y_val, has_data) if active]
    
    if active_models:
        # Filter for active models
        active_indices = [i for i, active in enumerate(has_data) if active]
        X_val_active = X_val[active_indices]
        y_val_active = [y_val[i] for i in active_indices]
        
        # Evaluate all classifiers
        classifiers = [
            ("Random Forest", clf),
            ("Decision Tree", clf2),
            ("Logistic Regression", clf3),
            ("SVC (RBF kernel)", clf4),
            ("SVC (Linear kernel)", clf5),
            ("SVC (Polynomial kernel)", clf6),
            ("Neural Network", clf7)
        ]
        
        for name, classifier in classifiers:
            y_pred = classifier.predict(X_val_active)
            accuracy = accuracy_score(y_val_active, y_pred)
            
            print(f"\nValidation results for {name}:")
            print(f"- Active models: {len(active_models)} out of {len(y_val)}")
            print(f"- Accuracy: {accuracy:.4f}")
            print(f"- Detailed report:")
            print(classification_report(y_val_active, y_pred))
    else:
        print("No active models in validation data (all zeros)")
    
    # Return all classifiers
    return [clf, clf2, clf3, clf7], train_data, X_val, y_val

def predict_model(clf, word_frequencies, clf_name=""):
    """
    Predict the model based on word frequencies
    
    Args:
        clf: Trained classifier
        word_frequencies: Dict of {word: frequency} for the new sample
        clf_name: Name of the classifier (for printing)
    
    Returns:
        Predicted model name
    """
    # Create a features array aligned with training data
    features = np.zeros((1, len(TRAINING_WORDS_LIST)))
    
    # Normalize the input frequencies
    total_freq = sum(word_frequencies.values())
    if total_freq == 0:
        print("Warning: Empty word frequencies provided")
        return None
        
    # Fill in the features array
    for word, freq in word_frequencies.items():
        if word in TRAINING_WORDS_DICT:
            idx = TRAINING_WORDS_DICT[word]
            features[0, idx] = freq / total_freq
    
    # Make prediction
    prediction = clf.predict(features)
    
    # Get probabilities if the classifier supports predict_proba
    if hasattr(clf, 'predict_proba'):
        probabilities = clf.predict_proba(features)
        
        # Get the top 3 most likely models
        top_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_models = [clf.classes_[i] for i in top_indices]
        top_probs = [probabilities[0][i] for i in top_indices]
        
        print(f"Predicted model using {clf_name}: {prediction[0]}")
        print("Top 3 predictions:")
        for model, prob in zip(top_models, top_probs):
            print(f"- {model}: {prob:.4f}")
    else:
        # For models without probability estimates
        print(f"Predicted model using {clf_name}: {prediction[0]}")
    
    return prediction[0]

def main():
    # File paths
    train_file_path = 'classifiers/heatmap_data.json'
    validation_file_path = 'classifiers/test/heatmap_data_3.json'
    
    # Train on one dataset and validate on another
    lst_classifiers, train_data, X_val, y_val = train_and_validate(train_file_path, validation_file_path)
    
    if lst_classifiers is None:
        print("Failed to train classifier. Exiting.")
        return
    
    # Example: Predict for a new set of word frequencies
    print("\nExample prediction:")
    new_word_frequencies = {
        'turbulent': 10,
        'mountainous': 5,
        'majestic': 3,
        'aquatic': 8,
        'separated': 6,
        'complex': 1,
        'ancient': 1
    }
    
    classifier_names = ["Random Forest", "Decision Tree", "Logistic Regression", "Neural Network"]
    
    for clf, name in zip(lst_classifiers, classifier_names):
        predicted_model = predict_model(clf, new_word_frequencies, name)


if __name__ == "__main__":
    main()