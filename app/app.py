from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Constants for model prediction (copied from heatMap_prelim_classifier.py)
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

# Create a word-to-index mapping for fast lookups
TRAINING_WORDS_DICT = {word: idx for idx, word in enumerate(TRAINING_WORDS_LIST)}

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

# Load the trained MLPClassifier model
def load_model():
    try:
        model_path = os.path.join('classifiers', 'mlp_classifier.pkl')
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        print("Model loaded successfully!")
        return classifier
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # If model not found in primary location, try alternative locations
        try:
            # Try relative path from app directory
            alt_path = os.path.join('..', 'classifiers', 'mlp_classifier.pkl')
            with open(alt_path, 'rb') as f:
                classifier = pickle.load(f)
            print("Model loaded successfully from alternative path!")
            return classifier
        except Exception as alt_e:
            print(f"Error loading model from alternative path: {str(alt_e)}")
            return None

# Initialize the model
classifier = load_model()

# Process word frequencies and prepare features for prediction
def prepare_features(word_frequencies):
    """
    Prepare feature vector from word frequencies dict for model prediction

    Args:
        word_frequencies: Dict of {word: frequency}

    Returns:
        Numpy array of features aligned with training data
    """
    # Create a features array aligned with training data
    features = np.zeros((1, len(TRAINING_WORDS_LIST)))

    # Normalize the input frequencies
    total_freq = sum(word_frequencies.values())
    if total_freq == 0:
        print("Warning: Empty word frequencies provided")
        return features

    # Fill in the features array
    for word, freq in word_frequencies.items():
        if word in TRAINING_WORDS_DICT:
            idx = TRAINING_WORDS_DICT[word]
            features[0, idx] = freq / total_freq

    # Check if we have any non-zero features
    if np.sum(features) == 0:
        print("Warning: No overlapping words between input and training set")

    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/identify-model', methods=['POST'])
def identify_model():
    """
    Endpoint to identify model based on word frequencies.

    Expected JSON input:
    {
        "api_key": "your_api_key",
        "provider": "provider_name",
        "word_frequencies": {"word1": freq1, "word2": freq2, ...}
    }
    """
    data = request.json
    api_key = data.get('api_key')
    provider = data.get('provider')
    word_frequencies = data.get('word_frequencies', {})

    if not api_key or not provider:
        return jsonify({"error": "Missing API key or provider"}), 400

    if not word_frequencies:
        return jsonify({"error": "Missing word frequencies data"}), 400

    if classifier is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    try:
        # Prepare features for prediction
        features = prepare_features(word_frequencies)

        # Make prediction
        predicted_model = classifier.predict(features)[0]

        # Get probabilities if available
        confidence = 0.0
        top_predictions = []

        if hasattr(classifier, 'predict_proba'):
            probabilities = classifier.predict_proba(features)[0]

            # Get indices sorted by probability (descending)
            top_indices = np.argsort(probabilities)[-5:][::-1]

            # Extract top 5 models and their probabilities
            top_models = [classifier.classes_[i] for i in top_indices]
            top_probs = [float(probabilities[i]) for i in top_indices]

            # Calculate confidence level
            confidence = top_probs[0]

            # Create top predictions list
            top_predictions = [
                {"model": model, "probability": float(prob)}
                for model, prob in zip(top_models, top_probs)
            ]

        # Prepare response
        model_info = {
            "provider": provider,
            "predicted_model": predicted_model,
            "confidence": f"{confidence:.2%}",
            "confidence_value": float(confidence),
            "top_predictions": top_predictions
        }

        return jsonify(model_info)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Endpoint to return the list of models the classifier knows about"""
    return jsonify({"models": LIST_OF_MODELS})

if __name__ == '__main__':
    app.run(debug=True)
