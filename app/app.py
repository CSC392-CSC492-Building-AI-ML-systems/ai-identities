from flask_socketio import SocketIO
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import re
import threading
import time
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler # Keep if needed for local, but careful in serverless
import sys
import sklearn
from openai import APIStatusError, APIConnectionError, RateLimitError, AuthenticationError, PermissionDeniedError, NotFoundError, OpenAI

app = Flask(__name__)
# Add a secret key for session management if needed, and for SocketIO security
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'your_secret_key_here!')
# Initialize SocketIO
# Using eventlet for async operations is generally recommended
socketio = SocketIO(app, async_mode='eventlet')

# --- Logging Configuration ---
# (Keep your existing logging setup, ensuring it works in your target environment)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
for handler in app.logger.handlers[:]:
    app.logger.removeHandler(handler)
app.logger.addHandler(console_handler)
log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
app.logger.setLevel(log_level)
app.logger.info(f"Logging configured. Level: {log_level_name}. Handler: Console.")
app.logger.info("Flask-SocketIO initialized.") # Add this line
# --- End Logging Configuration ---


# --- Constants ---
# (Keep TRAINING_WORDS_LIST, TRAINING_WORDS_DICT, LIST_OF_MODELS as they are)
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
TRAINING_WORDS_DICT = {word: idx for idx, word in enumerate(TRAINING_WORDS_LIST)}
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

# --- Provider Base URL Mapping ---
# Map provider names (lowercase) to their OpenAI-compatible base URLs
PROVIDER_BASE_URLS = {
    "openai": None, # Standard OpenAI API uses default base URL
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1", # Native API requires special headers and structure
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "mistral": "https://api.mistral.ai/v1/"
    # Add other providers here if they offer an OpenAI-compatible endpoint
}

# --- Load Model ---
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "mlp_classifier.pkl")
    try:
        app.logger.info(f"Attempting to load model from: {model_path}")
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        app.logger.info("Model loaded successfully!")
        return classifier
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return None

classifier = load_model()

# --- Feature Preparation ---
# (Keep prepare_features function as it is)
def prepare_features(word_frequencies):
    features = np.zeros((1, len(TRAINING_WORDS_LIST)))
    total_freq = sum(word_frequencies.values())
    if total_freq == 0:
        app.logger.warning("Empty word frequencies provided to prepare_features")
        return features
    for word, freq in word_frequencies.items():
        if word in TRAINING_WORDS_DICT:
            idx = TRAINING_WORDS_DICT[word]
            features[0, idx] = freq / total_freq
    app.logger.debug(f"Prepared features. Total frequency: {total_freq}")
    if np.sum(features) == 0:
        app.logger.warning("No overlapping words found between input and training set")
    return features


# --- Modified query_llm ---
def query_llm(api_key, provider, model, num_samples=100, batch_size=10, temperature=0.7,
              # Add socketio instance and client session ID
              socketio_instance=None, client_sid=None):
    """
    Query the LLM using the OpenAI SDK, emitting progress via SocketIO.
    Propagates critical errors by raising ProviderAPIError.

    Args:
        # ... (existing args) ...
        socketio_instance: The Flask-SocketIO instance.
        client_sid: The unique session ID of the client requesting the operation.

    Returns:
        List of responses from the LLM.

    Raises:
        ProviderAPIError: If a critical, non-transient error occurs during API calls.
    """
    responses = []
    prompt = "Describe the earth using only 10 adjectives. You can only use ten words, each separated by a comma."

    # Validate and adjust parameters
    temperature = max(0.0, min(temperature, 2.0))
    num_samples = max(10, min(num_samples, 4000))

    provider_lower = provider.lower()
    base_url = PROVIDER_BASE_URLS.get(provider_lower)
    default_headers = {}

    if provider_lower == 'anthropic':
        default_headers["anthropic-version"] = "2023-06-01"
        app.logger.warning("Anthropic call via OpenAI client may require a compatible endpoint.")
    if provider_lower == 'google':
        app.logger.warning("Google call via OpenAI client may need custom API structure.")

    def emit_progress(current, total, message):
        if socketio_instance and client_sid:
            try:
                socketio_instance.emit('progress', {
                    'current': current,
                    'total': total,
                    'message': message
                }, room=client_sid)
                app.logger.debug(f"Progress to {client_sid}: {current}/{total} - {message}")
            except Exception as e:
                app.logger.error(f"Emit failed to {client_sid}: {e}", exc_info=False)

    emit_progress(0, num_samples, f"Starting queries for {provider}/{model}...")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        timeout=15,
        max_retries=10
    )

    try:
        for i in range(num_samples):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=100,
                )
                content = response.choices[0].message.content if response.choices and response.choices[0].message else None
                if content:
                    responses.append(content)
                else:
                    app.logger.warning(f"[{i+1}] Empty response from {provider_lower}/{model}")
            except Exception as e:
                status_code = getattr(e, 'status_code', 500)
                is_critical = isinstance(e, (
                    AuthenticationError, PermissionDeniedError,
                    NotFoundError, APIStatusError
                )) and not getattr(e, 'status_code', 500) in (429, 500)

                error_msg = f"[{i+1}] Error from {provider_lower}/{model}: {type(e).__name__} - {str(e)[:200]}"
                if is_critical:
                    app.logger.error(f"Critical: {error_msg}")
                    emit_progress(i, num_samples, f"Error: {error_msg}")
                    raise ProviderAPIError(error_msg, status_code=status_code, provider=provider_lower, model=model)
                else:
                    app.logger.warning(f"Non-critical: {error_msg}")

            emit_progress(i + 1, num_samples, f"Querying {provider_lower}/{model}...")

            if (i + 1) % batch_size == 0:
                time.sleep(0.15)
            elif (i + 1) % 5 == 0:
                time.sleep(0.05)

        emit_progress(num_samples, num_samples, "Processing responses...")

    finally:
        try:
            client.close()
        except Exception as e:
            app.logger.warning(f"Error closing client: {e}", exc_info=False)

    app.logger.info(f"Finished {len(responses)} responses for SID {client_sid}")
    return responses


# --- Response Processing ---
def process_responses(responses):
    all_words = []
    processed_count = 0
    skipped_count = 0
    app.logger.info(f"Processing {len(responses)} raw responses...")

    for i, response in enumerate(responses):
        if not isinstance(response, str) or not response.strip():
            app.logger.debug(f"Skipping invalid/empty response at index {i}: Type={type(response)}, Value='{str(response)[:50]}...'")
            skipped_count += 1
            continue

        processed_count += 1
        words_text = response.lower().strip()
        words_text = re.sub(r'^(here are|okay,? here are|sure,? here are|here\'s a list of)?\s*(\d+\s+)?(adjectives|words)?\s*[:\-]*\s*', '', words_text, flags=re.IGNORECASE)
        words_text = re.sub(r'^\s*\d+\.?\s*', '', words_text)
        words_text = re.sub(r'\s*\d+\.?\s*$', '', words_text)
        words_text = re.sub(r'\s*\d+[\.\)]\s*', ',', words_text)
        words_text = re.sub(r'[\n;]+', ',', words_text)

        extracted_in_response = 0
        potential_words = words_text.split(',')
        for word in potential_words:
            cleaned_word = word.strip()
            cleaned_word = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned_word) # Punctuation from ends only
            if cleaned_word and len(cleaned_word) < 30:
                all_words.append(cleaned_word)
                extracted_in_response += 1
            elif cleaned_word:
                app.logger.debug(f"Skipping potentially invalid word: '{cleaned_word[:50]}...'")

        if extracted_in_response == 0 and response.strip():
            app.logger.debug(f"Response '{response[:50]}...' yielded no words after cleaning.")

    word_frequencies = Counter(all_words)
    app.logger.info(f"Processed {processed_count} valid responses (skipped {skipped_count}). Found {len(word_frequencies)} unique words. Total word occurrences: {len(all_words)}")
    if word_frequencies:
        top_5 = word_frequencies.most_common(5)
        app.logger.debug(f"Top 5 words: {top_5}")

    return word_frequencies

# --- Flask Routes ---

@app.route('/')
def home():
    app.logger.info("Serving home page")
    # You could potentially store the sid in the Flask session if needed elsewhere
    # session['user_sid'] = request.sid # Requires app.config['SECRET_KEY']
    return render_template('index.html')

# --- Modified identify_model route ---
@app.route('/api/identify-model', methods=['POST'])
def identify_model():
    start_time = time.time()

    if not request.is_json:
        # Log without SID here as we don't have it yet
        app.logger.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    # --- Get client_sid from the POST request body ---
    client_sid = data.get('client_sid')

    if not client_sid:
        app.logger.error("Identify request received without 'client_sid' in JSON payload.")
        # Progress updates are essential here, so fail if no SID provided
        return jsonify({"error": "Missing 'client_sid' in request payload"}), 400

    app.logger.info(f"Received identify request from SID: {client_sid}") # Now log with SID

    # ... (get api_key, provider, model, etc. from data as before) ...
    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    num_samples = data.get('num_samples', 100)
    temperature = data.get('temperature', 0.7)
    batch_size = data.get('batch_size', 10)

    # ... (Input Validation remains the same) ...
    try: num_samples = int(num_samples); num_samples = min(max(num_samples, 10), 4000)
    except: num_samples = 100
    try: temperature = float(temperature); temperature = max(0.0, min(temperature, 2.0))
    except: temperature = 0.7
    try: batch_size = int(batch_size); batch_size = min(max(batch_size, 1), 20)
    except: batch_size = 10

    log_api_key_snippet = f"{api_key[:4]}..." if api_key and len(api_key) > 4 else "Provided" if api_key else "None"
    app.logger.info(f"Identify request params for SID {client_sid}: Provider={provider}, Model={model}, Samples={num_samples}, Temp={temperature}, Batch={batch_size}, APIKey={log_api_key_snippet}")


    if not api_key or not provider or not model:
        # Add SID to error log if available
        app.logger.warning(f"Missing API key, provider, or model for SID {client_sid}")
        return jsonify({"error": "Missing API key, provider, or model"}), 400
    if classifier is None:
        return jsonify({"error": "Classifier model not loaded."}), 500
    if provider.lower() not in PROVIDER_BASE_URLS:
         app.logger.warning(f"Provider '{provider}' not explicitly listed for SID {client_sid}. Attempting with default OpenAI base URL.")

    try:
        app.logger.info(f"Starting LLM query task for SID {client_sid} ({provider}/{model})...")
        responses = query_llm(
            api_key, provider, model, num_samples, batch_size, temperature,
            socketio_instance=socketio, client_sid=client_sid # Pass the correct SID
        )

        if not responses:
            # query_llm should have emitted progress/errors, but we still return HTTP error
            return jsonify({"error": f"Failed to collect responses from {provider}/{model}. Check logs/provider status."}), 500
        app.logger.info(f"Collected {len(responses)} responses for SID {client_sid}.")

        # --- Processing and Prediction --- (Remains largely the same)
        word_frequencies = process_responses(responses)
        if not word_frequencies:
            return jsonify({"error": "No valid words extracted from responses."}), 400
        app.logger.info(f"Extracted {len(word_frequencies)} unique words for SID {client_sid}.")

        features = prepare_features(word_frequencies)
        if np.sum(features) == 0: app.logger.warning(f"Feature vector is all zeros for SID {client_sid}.")

        app.logger.info(f"Making prediction for SID {client_sid}...")
        # ... (prediction logic remains the same) ...
        raw_prediction = classifier.predict(features)[0]
        prediction = raw_prediction
        if isinstance(raw_prediction, (np.integer, np.int64)): prediction = int(raw_prediction)
        elif hasattr(raw_prediction, 'item'): prediction = raw_prediction.item()

        class_labels = getattr(classifier, 'classes_', []).tolist()
        predicted_model = "unknown"
        predicted_index = -1
        # ... (logic to determine predicted_model and index) ...
        if class_labels:
             if isinstance(prediction, str) and prediction in class_labels:
                 predicted_model = prediction
                 try: predicted_index = class_labels.index(prediction)
                 except ValueError: predicted_index = -1
             elif isinstance(prediction, int) and 0 <= prediction < len(class_labels):
                 predicted_model = class_labels[prediction]
                 predicted_index = prediction
             else: app.logger.warning(f"Prediction '{prediction}' type {type(prediction)} invalid for SID {client_sid}.")
        else: app.logger.error("Cannot determine model name: class labels unavailable.")

        app.logger.info(f"Predicted Model='{predicted_model}', Index={predicted_index} for SID {client_sid}")

        confidence = 0.0
        top_predictions = []
        # ... (confidence calculation remains the same) ...
        if hasattr(classifier, 'predict_proba') and class_labels and predicted_index != -1:
             try:
                 probabilities = classifier.predict_proba(features)[0]
                 if len(probabilities) == len(class_labels):
                     sorted_indices = np.argsort(probabilities)[::-1]
                     top_predictions = [{"model": class_labels[i], "probability": float(probabilities[i])} for i in sorted_indices[:5]]
                     confidence = float(probabilities[predicted_index])
                     preds_log = [{'m': p['model'], 'p': f"{p['probability']:.4f}"} for p in top_predictions]
                     app.logger.info(f"Confidence: {confidence:.4f}, Top 5: {preds_log} for SID {client_sid}")
                 else: app.logger.error(f"Probability length mismatch for SID {client_sid}")
             except Exception as proba_error: app.logger.error(f"Error getting probabilities for SID {client_sid}: {proba_error}", exc_info=False)

        end_time = time.time()
        duration = end_time - start_time
        app.logger.info(f"Identify request for SID {client_sid} ({provider}/{model}) completed in {duration:.2f} seconds.")

        # --- Prepare final response ---
        status_message = "success"
        final_predicted_model = predicted_model
        if predicted_model == "unknown": status_message = "success_unrecognized"; final_predicted_model = "unrecognized_model"
        elif np.sum(features) == 0: status_message = "success_no_overlap"

        # Emit final 'complete' state via socket? Optional, as HTTP response signals end.
        socketio.emit('progress', {'current': num_samples, 'total': num_samples, 'message': 'Complete!'}, room=client_sid)

        model_info = {
            "provider": provider, "input_model": model, "samples_collected": len(responses),
            "unique_words_extracted": len(word_frequencies), "predicted_model": final_predicted_model,
            "confidence": f"{confidence:.2%}", "confidence_value": confidence, "top_predictions": top_predictions,
            "word_frequencies_top": dict(word_frequencies.most_common(20)), "status": status_message,
            "processing_time_seconds": round(duration, 2)
        }
        return jsonify(model_info)

    # --- Error Handling remains the same, using the extracted client_sid
    except ProviderAPIError as e: # Catch critical API errors propagated from query_llm
        duration = time.time() - start_time
        app.logger.error(f"ProviderAPIError during identify request for SID {client_sid} ({duration:.2f}s): {e}", exc_info=False) # Don't need full trace for this
        # Progress update for error should have been emitted in query_llm
        return jsonify({
            "error": f"API Error: {e.message} (Provider: {e.provider}, Model: {e.model}, Status: {e.status_code})",
            "status": "error_api",
            "processing_time_seconds": round(duration, 2)
        }), getattr(e, 'status_code', 500) if isinstance(getattr(e, 'status_code', None), int) and getattr(e, 'status_code', 500) >= 400 else 500 # Return provider status code if client error

    except Exception as e: # Catch other unexpected errors
        duration = time.time() - start_time
        app.logger.error(f"Unexpected error during identify request for SID {client_sid} ({duration:.2f}s): {str(e)}", exc_info=True)
        # Try to emit a final error state if possible
        if client_sid:
             socketio.emit('progress', {'current': 0, 'total': 1, 'message': f'Error: {str(e)}'}, room=client_sid)
        return jsonify({
            "error": f"An unexpected server error occurred: {str(e)}",
            "status": "error_internal",
            "processing_time_seconds": round(duration, 2)
        }), 500


@app.route('/api/models', methods=['GET'])
# ... (get_models remains the same) ...
def get_models():
    app.logger.info("Request received for /api/models")
    known_models = LIST_OF_MODELS # Default
    if classifier and hasattr(classifier, 'classes_'):
        classifier_classes = classifier.classes_.tolist()
        if len(classifier_classes) > 0:
            known_models = classifier_classes
            if set(LIST_OF_MODELS) != set(classifier_classes):
                app.logger.warning("Mismatch between hardcoded LIST_OF_MODELS and classifier.classes_!")

    supported_providers = list(PROVIDER_BASE_URLS.keys())
    return jsonify({
        "models": known_models,
        "supported_providers": supported_providers
    })

@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    start_time = time.time()
    # Get SID from payload here too, although it might be less critical for test
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    client_sid = data.get('client_sid') # Get SID from payload

    if client_sid:
        app.logger.info(f"Received test connection request from SID: {client_sid}")
    else:
        # Test might proceed without SID if progress isn't needed/sent
        app.logger.info("Received test connection request (no SID provided).")

    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    temperature = data.get('temperature', 0.1)

    try: temperature = max(0.0, min(float(temperature), 1.0))
    except: temperature = 0.1

    log_api_key_snippet = f"{api_key[:4]}..." if api_key and len(api_key) > 4 else "Provided" if api_key else "None"
    app.logger.info(f"Test connection params for SID {client_sid or 'N/A'}: Provider={provider}, Model={model}, Temp={temperature}, APIKey={log_api_key_snippet}")


    if not api_key or not provider or not model:
        return jsonify({"error": "Missing API key, provider, or model"}), 400

    # ... (rest of the try/except block) ...
    # Note: query_llm is called without socketio_instance/client_sid here,
    # which is fine for the test connection as it doesn't need progress updates.
    try:
        app.logger.info(f"Attempting single query to test connection to {provider}/{model} for SID {client_sid or 'N/A'}")
        responses = query_llm(api_key, provider, model, num_samples=1, temperature=temperature, batch_size=1)
        duration = time.time() - start_time

        if responses and isinstance(responses[0], str) and responses[0].strip():
            app.logger.info(f"Test connection successful for {provider}/{model} (SID {client_sid}, {duration:.2f}s).")
            return jsonify({
                "status": "success",
                "message": f"Successfully connected to {provider} and received response from {model}.",
                "response_preview": responses[0][:100] + ('...' if len(responses[0]) > 100 else ''),
                "processing_time_seconds": round(duration, 2)
            })
        elif responses:
            app.logger.warning(f"Test connection for SID {client_sid} to {provider}/{model} returned an invalid/empty response: Type={type(responses[0])} ({duration:.2f}s)")
            return jsonify({"status": "error", "message": f"Connected to {provider}/{model} but received an empty or invalid response.", "response_type": str(type(responses[0])), "processing_time_seconds": round(duration, 2)}), 500
        else:
            app.logger.warning(f"Test connection failed for SID {client_sid}: No response from {provider}/{model} ({duration:.2f}s).")
            # If query_llm raises ProviderAPIError, it will be caught below.
            # This 'else' covers cases where query_llm runs but returns empty list due to non-critical errors.
            return jsonify({"status": "error", "message": f"Failed to get response from '{model}' via '{provider}'. Possible non-critical errors (check server logs).", "processing_time_seconds": round(duration, 2)}), 500

    except ProviderAPIError as e: # Catch critical errors from the single query
        duration = time.time() - start_time
        app.logger.error(f"Test connection ProviderAPIError for SID {client_sid} ({provider}/{model}, {duration:.2f}s): {e}", exc_info=False)
        return jsonify({
            "status": "error",
            "message": f"API Error: {e.message}", # Simplified message for test
            "processing_time_seconds": round(duration, 2)
        }), getattr(e, 'status_code', 500) if isinstance(getattr(e, 'status_code', None), int) and getattr(e, 'status_code', 500) >= 400 else 500
    except Exception as e: # Catch other unexpected errors
        duration = time.time() - start_time
        app.logger.error(f"Test connection unexpected error for SID {client_sid} ({provider}/{model}, {duration:.2f}s): {str(e)}", exc_info=True)
        error_message = f"An unexpected error occurred: {str(e)}"
        return jsonify({"status": "error", "message": error_message, "processing_time_seconds": round(duration, 2)}), 500


# --- SocketIO Event Handlers (Example: connect/disconnect) ---
@socketio.on('connect')
def handle_connect():
    # request.sid is the unique ID for the client connection
    app.logger.info(f'Client connected: {request.sid}')
    # You could emit a welcome message or initial state if needed
    # emit('status', {'message': 'Connected to server'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    app.logger.info(f'Client disconnected: {request.sid}')


# --- Custom Exception for API Errors ---
# ... (ProviderAPIError class remains the same) ...
class ProviderAPIError(Exception):
    """Custom exception for critical API errors from providers."""
    def __init__(self, message, status_code=None, provider=None, model=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.provider = provider
        self.model = model

    def __str__(self):
        return f"ProviderAPIError(provider={self.provider}, model={self.model}, status={self.status_code}): {self.message}"


# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # Debug mode with Flask dev server might cause issues with socketio/eventlet reloading
    # For development, running without debug/reloader might be more stable,
    # or use specific run configurations suggested by Flask-SocketIO docs.
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    use_reloader = debug_mode # Generally okay with eventlet, but monitor
    app.logger.info(f"Starting Flask-SocketIO application with eventlet. Debug mode: {debug_mode}, Port: {port}")
    # Use socketio.run to start the server correctly
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port, use_reloader=use_reloader)
