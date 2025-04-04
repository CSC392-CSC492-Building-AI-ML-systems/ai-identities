# --- START OF REFACTORED FILE app.py ---

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
import openai # Use the OpenAI library exclusively

app = Flask(__name__)

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


def query_llm(api_key, provider, model, num_samples=100, batch_size=10, temperature=0.7):
    """
    Query the LLM using the OpenAI SDK, configuring base_url for different providers.

    Args:
        api_key: API key for the provider
        provider: Provider name (e.g., 'openai', 'anthropic', 'google', 'deepinfra', 'mistral')
        model: Model identifier to query
        num_samples: Number of samples to collect (10-1000)
        batch_size: Number of requests to send in parallel (1-20)
        temperature: Controls randomness (0=deterministic, 2=most random)

    Returns:
        List of responses from the LLM
    """
    responses = []
    prompt = "Describe the earth using only 10 adjectives. You can only use ten words, each separated by a comma."

    # Validate and adjust parameters
    temperature = max(0.0, min(temperature, 2.0))
    num_samples = max(10, min(num_samples, 1000))
    batch_size = max(1, min(batch_size, 20)) # Consider adjusting based on provider limits

    provider_lower = provider.lower()
    base_url = PROVIDER_BASE_URLS.get(provider_lower)

    app.logger.info(f"Starting LLM query: Provider={provider_lower}, Model={model}, Samples={num_samples}, BatchSize={batch_size}, Temp={temperature}, BaseURL={base_url or 'Default OpenAI'}")

    # --- Provider Specific Configuration ---
    # Anthropic requires specific headers when using its native API via an OpenAI client proxy/adapter
    # Note: Directly hitting api.anthropic.com/v1 with the standard openai client is unlikely to work
    # without a translation layer or using Anthropic's specific SDK. This assumes either a proxy
    # or that the user knows this might fail for standard Anthropic endpoints.
    default_headers = {}
    api_params = {} # Extra params for the create call if needed

    if provider_lower == 'anthropic':
        # Standard Anthropic requires this header. Add if using a proxy that expects it.
        default_headers["anthropic-version"] = "2023-06-01"
        # Anthropic uses 'max_tokens_to_sample' not 'max_tokens' in its native API
        # If using a proxy that translates, 'max_tokens' might work.
        # If hitting native endpoint via adapter, might need custom logic.
        # For simplicity here, we'll *assume* 'max_tokens' works via the chosen base_url.
        app.logger.warning("Attempting Anthropic call via OpenAI client. Requires compatible endpoint/proxy. Using standard 'max_tokens' parameter.")
        # api_params["max_tokens_to_sample"] = 100 # Example if native param needed

    if provider_lower == 'google':
        # The specified Google base URL might expect a different API structure (generateContent)
        # and model name format (e.g., "models/gemini-1.5-flash-latest").
        # This might fail if 'model' is passed as 'gemini-1.5-flash'.
        app.logger.warning("Attempting Google call via OpenAI client. Base URL might expect different API structure/model format (e.g., 'models/gemini...').")
        # Google might use 'maxOutputTokens' instead of 'max_tokens'.
        # Adjust model name if needed based on proxy/endpoint behavior:
        # if not model.startswith("models/"):
        #     model = f"models/{model}" # Example adaptation

    # Use locks for thread-safe list append
    lock = threading.Lock()
    threads = []

    def send_single_request(attempt_index):
        nonlocal responses # Ensure we modify the outer scope variable
        # Instantiate client inside the thread to ensure config is isolated
        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
                timeout=5, # Set a request timeout
            )
        except Exception as client_err:
            app.logger.error(f"Failed to initialize OpenAI client for {provider_lower}: {client_err}", exc_info=False)
            return # Cannot proceed

        try:
            request_start_time = time.time()
            response_obj = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=100, # Standard OpenAI param
                # **api_params # Add provider-specific params if needed
            )
            request_duration = time.time() - request_start_time
            app.logger.debug(f"Request {attempt_index+1} to {provider_lower}/{model} completed in {request_duration:.2f}s")

            content = None
            if response_obj.choices and response_obj.choices[0].message:
                content = response_obj.choices[0].message.content

            if content:
                with lock:
                    responses.append(content)
                # Success
            else:
                app.logger.warning(f"Request {attempt_index+1} to {provider_lower}/{model} yielded empty content. Response: {response_obj}")
                # Consider this a failure for this attempt

        # --- Error Handling (Single Attempt) ---
        except openai.AuthenticationError as e:
            app.logger.error(f"OpenAI Authentication Error for request {attempt_index+1} ({provider_lower}/{model}): {e}. Check API key.", exc_info=False)
        except openai.PermissionDeniedError as e:
            app.logger.error(f"OpenAI Permission Denied Error for request {attempt_index+1} ({provider_lower}/{model}): {e}. Check API key permissions/model access.", exc_info=False)
        except openai.NotFoundError as e:
            app.logger.error(f"OpenAI Not Found Error for request {attempt_index+1} ({provider_lower}/{model}): {e}. Check model name and base URL.", exc_info=False)
        except openai.RateLimitError as e:
            app.logger.error(f"OpenAI Rate Limit Error for request {attempt_index+1} ({provider_lower}/{model}): {e}. Request failed.", exc_info=False)
        except openai.APIConnectionError as e:
            app.logger.error(f"OpenAI API Connection Error for request {attempt_index+1} ({provider_lower}/{model}): {e}. Request failed.", exc_info=False)
        except openai.APIStatusError as e: # Catch other API errors (e.g., 500s)
             app.logger.error(f"OpenAI API Status Error {e.status_code} for request {attempt_index+1} ({provider_lower}/{model}): {e}. Request failed.", exc_info=False)
        except Exception as e:
            # Catch any other unexpected errors during the API call
            app.logger.error(f"Unexpected error during API call for request {attempt_index+1} ({provider_lower}/{model}): {type(e).__name__} - {e}", exc_info=True)

        # No retry logic here, the function simply completes after one attempt (success or logged error)

    # --- Threading Logic ---
    try:
        for i in range(num_samples):
            thread = threading.Thread(target=send_single_request, args=(i,))
            thread.start()
            threads.append(thread)

            # Simple batching delay
            if (i + 1) % batch_size == 0:
                app.logger.debug(f"Launched batch of {batch_size} threads (up to {i+1}/{num_samples}). Pausing briefly...")
                time.sleep(0.2) # Adjust delay as needed based on provider rate limits

            # Optional smaller stagger within a batch
            elif (i+1) % 5 == 0:
                time.sleep(0.05)

        # Wait for all threads to complete
        for i, thread in enumerate(threads):
            thread.join(timeout=90) # Generous timeout per thread
            if thread.is_alive():
                app.logger.warning(f"Thread {i+1} for {provider_lower}/{model} timed out.")

        app.logger.info(f"Finished {provider_lower} query for {model}. Attempted {num_samples} samples, successfully collected {len(responses)} responses.")

    except Exception as e:
        app.logger.error(f"Critical error during threading or request dispatch for {provider_lower}/{model}: {str(e)}", exc_info=True)
        # Return partial results if any collected
        return responses

    return responses


# --- Response Processing ---
# (Keep process_responses function as it is)
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
    return render_template('index.html')

@app.route('/api/identify-model', methods=['POST'])
def identify_model():
    start_time = time.time()
    app.logger.info("Received request for /api/identify-model")
    if not request.is_json:
        app.logger.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    num_samples = data.get('num_samples', 100)
    temperature = data.get('temperature', 0.7)
    batch_size = data.get('batch_size', 10) # Allow specifying batch size

    # --- Input Validation ---
    try:
        num_samples = int(num_samples)
        num_samples = min(max(num_samples, 10), 1000)
    except (ValueError, TypeError):
        num_samples = 100
    try:
        temperature = float(temperature)
        temperature = max(0.0, min(temperature, 2.0))
    except (ValueError, TypeError):
        temperature = 0.7
    try:
        batch_size = int(batch_size)
        batch_size = min(max(batch_size, 1), 20) # Limit batch size
    except (ValueError, TypeError):
        batch_size = 10 # Default if invalid

    log_api_key_snippet = f"{api_key[:4]}..." if api_key and len(api_key) > 4 else "Provided" if api_key else "None"
    app.logger.info(f"Identify request params: Provider={provider}, Model={model}, Samples={num_samples}, Temp={temperature}, Batch={batch_size}, APIKey={log_api_key_snippet}")

    if not api_key or not provider or not model:
        return jsonify({"error": "Missing API key, provider, or model"}), 400
    if classifier is None:
        return jsonify({"error": "Classifier model not loaded."}), 500

    # Check if provider is known (optional but good)
    if provider.lower() not in PROVIDER_BASE_URLS:
        # We can still attempt the call if base_url is None (defaults to OpenAI)
        # Or we can return an error if it's explicitly unknown.
        # Let's allow attempting with default OpenAI URL if provider unknown.
        app.logger.warning(f"Provider '{provider}' not explicitly listed. Attempting with default OpenAI base URL.")
        # Alternatively, return error:
        # return jsonify({"error": f"Unsupported provider: {provider}. Supported: {list(PROVIDER_BASE_URLS.keys())}"}), 400


    try:
        app.logger.info(f"Starting LLM query task for {provider}/{model}...")
        responses = query_llm(api_key, provider, model, num_samples, batch_size, temperature)

        if not responses:
            return jsonify({"error": f"Failed to collect responses from {provider}/{model}. Check logs."}), 500
        app.logger.info(f"Collected {len(responses)} responses.")

        word_frequencies = process_responses(responses)
        if not word_frequencies:
            return jsonify({"error": "No valid words extracted from responses."}), 400
        app.logger.info(f"Extracted {len(word_frequencies)} unique words.")

        features = prepare_features(word_frequencies)
        if np.sum(features) == 0:
            app.logger.warning("Feature vector is all zeros (no overlap with training words).")

        app.logger.info("Making prediction...")
        raw_prediction = classifier.predict(features)[0]

        prediction = raw_prediction
        if isinstance(raw_prediction, (np.integer, np.int64)): prediction = int(raw_prediction)
        elif hasattr(raw_prediction, 'item'): prediction = raw_prediction.item()

        class_labels = []
        if hasattr(classifier, 'classes_'): class_labels = classifier.classes_.tolist()
        else: app.logger.warning("Classifier missing 'classes_' attribute.")

        predicted_model = "unknown"
        predicted_index = -1
        if class_labels:
            if isinstance(prediction, str) and prediction in class_labels:
                predicted_model = prediction
                try: predicted_index = class_labels.index(prediction)
                except ValueError: predicted_index = -1
            elif isinstance(prediction, int) and 0 <= prediction < len(class_labels):
                predicted_model = class_labels[prediction]
                predicted_index = prediction
            else:
                app.logger.warning(f"Prediction '{prediction}' type {type(prediction)} invalid or out of range.")
        else:
            app.logger.error("Cannot determine model name: class labels unavailable.")

        app.logger.info(f"Predicted Model='{predicted_model}', Index={predicted_index}")

        confidence = 0.0
        top_predictions = []
        if hasattr(classifier, 'predict_proba') and class_labels and predicted_index != -1:
            try:
                probabilities = classifier.predict_proba(features)[0]
                if len(probabilities) == len(class_labels):
                    sorted_indices = np.argsort(probabilities)[::-1]
                    top_predictions = [
                        {"model": class_labels[i], "probability": float(probabilities[i])}
                        for i in sorted_indices[:5]
                    ]
                    confidence = float(probabilities[predicted_index])
                    preds_log = [{'m': p['model'], 'p': f"{p['probability']:.4f}"} for p in top_predictions]
                    app.logger.info(f"Confidence: {confidence:.4f}, Top 5: {preds_log}")
                else:
                    app.logger.error(f"Probability array length mismatch ({len(probabilities)} vs {len(class_labels)})")
            except Exception as proba_error:
                app.logger.error(f"Error getting probabilities: {proba_error}", exc_info=False)

        end_time = time.time()
        duration = end_time - start_time
        app.logger.info(f"Identify request for {provider}/{model} completed in {duration:.2f} seconds.")

        status_message = "success"
        final_predicted_model = predicted_model
        if predicted_model == "unknown":
            status_message = "success_unrecognized"
            final_predicted_model = "unrecognized_model"
        elif np.sum(features) == 0:
            status_message = "success_no_overlap"

        model_info = {
            "provider": provider,
            "input_model": model,
            "samples_collected": len(responses),
            "unique_words_extracted": len(word_frequencies),
            "predicted_model": final_predicted_model,
            "confidence": f"{confidence:.2%}",
            "confidence_value": confidence,
            "top_predictions": top_predictions,
            "word_frequencies_top": dict(word_frequencies.most_common(20)),
            "status": status_message,
            "processing_time_seconds": round(duration, 2)
        }
        return jsonify(model_info)

    except Exception as e:
        duration = time.time() - start_time
        app.logger.error(f"Error during identify model request ({duration:.2f}s) for {provider}/{model}: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "status": "error",
            "processing_time_seconds": round(duration, 2)
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    app.logger.info("Request received for /api/models")
    known_models = LIST_OF_MODELS # Default
    if classifier and hasattr(classifier, 'classes_'):
        classifier_classes = classifier.classes_.tolist()
        if len(classifier_classes) > 0:
            known_models = classifier_classes
            if set(LIST_OF_MODELS) != set(classifier_classes):
                app.logger.warning("Mismatch between hardcoded LIST_OF_MODELS and classifier.classes_!")

    # Also return the providers we have base URLs for
    supported_providers = list(PROVIDER_BASE_URLS.keys())

    return jsonify({
        "models": known_models,
        "supported_providers": supported_providers # Inform frontend about providers
    })


@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    start_time = time.time()
    app.logger.info("Received request for /api/test-connection")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    temperature = data.get('temperature', 0.1) # Low temp for test

    try: temperature = max(0.0, min(float(temperature), 1.0))
    except: temperature = 0.1

    log_api_key_snippet = f"{api_key[:4]}..." if api_key and len(api_key) > 4 else "Provided" if api_key else "None"
    app.logger.info(f"Test connection params: Provider={provider}, Model={model}, Temp={temperature}, APIKey={log_api_key_snippet}")

    if not api_key or not provider or not model:
        return jsonify({"error": "Missing API key, provider, or model"}), 400

    # Check if provider is known (optional)
    if provider.lower() not in PROVIDER_BASE_URLS:
        app.logger.warning(f"Provider '{provider}' not explicitly listed for test. Attempting with default OpenAI base URL.")


    try:
        app.logger.info(f"Attempting single query to test connection to {provider}/{model}")
        # Use num_samples=1, batch_size=1, and handle retries within query_llm
        responses = query_llm(api_key, provider, model, num_samples=1, temperature=temperature, batch_size=1)
        app.logger.info(f"query_llm call for test connection returned: {responses}\n")
        duration = time.time() - start_time

        if responses and isinstance(responses[0], str) and responses[0].strip():
            app.logger.info(f"Test connection successful for {provider}/{model} ({duration:.2f}s).")
            return jsonify({
                "status": "success",
                "message": f"Successfully connected to {provider} and received response from {model}.",
                "response_preview": responses[0][:100] + ('...' if len(responses[0]) > 100 else ''),
                "processing_time_seconds": round(duration, 2)
            })
        elif responses:
            app.logger.warning(f"Test connection to {provider}/{model} returned an invalid/empty response: Type={type(responses[0])} ({duration:.2f}s)")
            return jsonify({
                "status": "error",
                "message": f"Connected to {provider}/{model} but received an empty or invalid response.",
                "response_type": str(type(responses[0])),
                "processing_time_seconds": round(duration, 2)
            }), 500
        else: # No response collected
            app.logger.warning(f"Test connection failed: No response from {provider}/{model} ({duration:.2f}s). Check API key, model name, provider status, base URL compatibility.")
            return jsonify({
                "status": "error",
                "message": f"Failed to get response from '{model}' via '{provider}'. Check API key, model name, Base URL, and provider status. See server logs for details.",
                "processing_time_seconds": round(duration, 2)
            }), 500

    except Exception as e:
        duration = time.time() - start_time
        app.logger.error(f"Test connection error for {provider}/{model} ({duration:.2f}s): {str(e)}", exc_info=True)
        # Check for specific OpenAI errors if possible to give better feedback
        error_message = f"An unexpected error occurred: {str(e)}"
        if isinstance(e, openai.AuthenticationError):
            error_message = "Authentication failed. Check your API key."
        elif isinstance(e, openai.PermissionDeniedError):
            error_message = "Permission denied. Check API key permissions or model access."
        elif isinstance(e, openai.NotFoundError):
            error_message = f"Model '{model}' not found or Base URL for '{provider}' is incorrect/incompatible."
        elif isinstance(e, openai.RateLimitError):
            error_message = "Rate limit exceeded. Please wait and try again."
        elif isinstance(e, openai.APIConnectionError):
            error_message = f"Could not connect to the API endpoint for '{provider}'. Check network or Base URL."

        return jsonify({
            "status": "error",
            "message": error_message,
            "processing_time_seconds": round(duration, 2)
        }), 500


# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.logger.info(f"Starting Flask application. Debug mode: {debug_mode}, Port: {port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port, use_reloader=debug_mode)

# --- END OF REFACTORED FILE app.py ---
