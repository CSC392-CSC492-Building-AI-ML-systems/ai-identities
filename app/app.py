from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import re
import threading
import time
import uuid
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler
import sys
import sklearn
from openai import APIStatusError, APIConnectionError, RateLimitError, AuthenticationError, PermissionDeniedError, NotFoundError, OpenAI

app = Flask(__name__)

# --- Task Management Store (In-Memory) ---
# WARNING: This in-memory store is simple but will be lost if the server restarts.
# For production, use a database, cache (like Redis), or a more robust task queue system.
tasks = {}
task_lock = threading.Lock() # To ensure thread-safe access to the tasks dictionary

# --- Logging Configuration ---
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


def query_llm(api_key, provider, model, num_samples=100, batch_size=10, temperature=0.7, progress_callback=None, task_id=None):
    """
    Query the LLM using the OpenAI SDK, configure base_url, and report progress.
    Propagates critical errors by raising ProviderAPIError.

    Args:
        api_key: API key for the provider
        provider: Provider name (e.g., 'openai', 'anthropic', 'google', 'deepinfra', 'mistral')
        model: Model identifier to query
        num_samples: Number of samples to collect (10-1000)
        batch_size: Number of parallel requests (1-20)
        temperature: Controls randomness
        progress_callback: Optional function to call with (completed_count, total_samples)
        task_id: Optional task ID for logging context

    Returns:
        List of responses from the LLM.

    Raises:
        ProviderAPIError: If a critical, non-transient error occurs during API calls (e.g., Auth, 422).
    """
    responses = []
    prompt = "What are the 15 best words to describe the Earth? Write only those words on one line, in order from highest ranked to lowest ranked, each separated by the symbol \"|\"."

    # Validate and adjust parameters
    temperature = max(0.0, min(temperature, 2.0))
    num_samples = max(10, min(num_samples, 10))
    batch_size = max(1, min(batch_size, 20))

    provider_lower = provider.lower()
    base_url = PROVIDER_BASE_URLS.get(provider_lower)

    log_prefix = f"[Task={task_id}] " if task_id else ""
    app.logger.info(f"{log_prefix}Starting LLM query: Provider={provider_lower}, Model={model}, Samples={num_samples}, BatchSize={batch_size}, Temp={temperature}, BaseURL={base_url or 'Default OpenAI'}")

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
        app.logger.warning(f"{log_prefix}Attempting Anthropic call via OpenAI client. Requires compatible endpoint/proxy.")
    if provider_lower == 'google':
        # The specified Google base URL might expect a different API structure (generateContent)
        # and model name format (e.g., "models/gemini-1.5-flash-latest").
        # This might fail if 'model' is passed as 'gemini-1.5-flash'.
        app.logger.warning(f"{log_prefix}Attempting Google call via OpenAI client. Base URL might expect different API structure/model format.")

    # Locks and shared state for this specific query batch
    response_lock = threading.Lock()
    progress_lock = threading.Lock() # Lock specifically for updating completed_count
    threads = []
    first_critical_error = None # Shared variable to store the first critical error
    completed_count = 0 # Track successful responses within this query_llm call

    def send_single_request(attempt_index):
        nonlocal responses, first_critical_error, completed_count # Allow modification of outer scope variables

        # Check if a critical error has already been flagged by another thread
        with response_lock:
            if first_critical_error is not None:
                return # Stop this thread early if another thread hit a critical error

        # --- Client Initialization ---
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
                timeout=5,
            )
        except Exception as client_err:
            msg = f"Failed to initialize API client: {client_err}"
            app.logger.error(f"{log_prefix}[Req {attempt_index+1}] {msg}", exc_info=False)
            # Consider this a critical configuration error
            with response_lock:
                if first_critical_error is None:
                    first_critical_error = ProviderAPIError(msg, provider=provider_lower, model=model)
            return

        # --- API Call and Error Handling ---
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
            app.logger.debug(f"{log_prefix}Request {attempt_index+1} to {provider_lower}/{model} completed in {request_duration:.2f}s")

            content = None
            if response_obj.choices and response_obj.choices[0].message:
                content = response_obj.choices[0].message.content

            if content:
                current_count_snapshot = 0
                with response_lock:
                    # Only add response if no critical error has occurred
                    if first_critical_error is None:
                        responses.append(content)
                        # Update progress safely *after* adding response
                        with progress_lock:
                            completed_count += 1
                            current_count_snapshot = completed_count # Get value under lock

                # Call callback *outside* locks if progress occurred
                if progress_callback and current_count_snapshot > 0:
                     try:
                         # Pass current count and the target number of samples
                         progress_callback(current_count_snapshot, num_samples)
                     except Exception as cb_err:
                         app.logger.error(f"{log_prefix}Error in progress callback: {cb_err}", exc_info=False)
            else:
                app.logger.warning(f"{log_prefix}[Req {attempt_index+1}] {provider_lower}/{model} yielded empty content.")

        # --- Error Handling (Mostly same, just ensure logs have log_prefix) ---
        # Authentication/Permission/NotFound errors
        except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
            error_msg_map = {
                AuthenticationError: f"Authentication Failed for {provider_lower}. Check API Key.",
                PermissionDeniedError: f"Permission Denied for {provider_lower}/{model}. Check permissions/access.",
                NotFoundError: f"Model '{model}' not found via {provider_lower} or Base URL is incorrect/incompatible."
            }
            status_code_map = { AuthenticationError: 401, PermissionDeniedError: 403, NotFoundError: 404 }
            error_msg = error_msg_map.get(type(e), "Authorization/Resource Error")
            status_code = status_code_map.get(type(e))
            app.logger.error(f"{log_prefix}[Req {attempt_index+1}] {error_msg}: {e}", exc_info=False)
            with response_lock:
                if first_critical_error is None:
                    first_critical_error = ProviderAPIError(error_msg, status_code=status_code, provider=provider_lower, model=model)

        # APIStatusError (422, 400, 429, 5xx, etc.)
        except APIStatusError as e:
            status_code = e.status_code
            error_msg = f"API Status Error {status_code} for {provider_lower}/{model}"
            is_critical = False
            if status_code == 422:
                detailed_message = f"Invalid request (422): {str(e)}"
                try:
                    error_body = e.response.json()
                    detailed_message = f"Invalid request (422): {error_body.get('message', str(e))}"
                except Exception: pass # Ignore parsing errors
                error_msg = detailed_message
                is_critical = True
            elif status_code == 400:
                error_msg = f"Bad Request (400) for {provider_lower}/{model}. Check parameters/request format."
                is_critical = True
            elif status_code == 429:
                error_msg = f"Rate Limit Exceeded (429) for {provider_lower}/{model}. Request failed."
                app.logger.warning(f"{log_prefix}[Req {attempt_index+1}] {error_msg}", exc_info=False)
                is_critical = False
            elif status_code >= 500:
                error_msg = f"Server Error ({status_code}) from {provider_lower}/{model}. Request failed."
                app.logger.warning(f"{log_prefix}[Req {attempt_index+1}] {error_msg}: {e}", exc_info=False)
                is_critical = False
            else: # Other client-side errors (4xx)
                error_msg = f"API Client Error ({status_code}) for {provider_lower}/{model}: {e}"
                app.logger.error(f"{log_prefix}[Req {attempt_index+1}] {error_msg}", exc_info=False)
                is_critical = True

            if is_critical:
                with response_lock:
                    if first_critical_error is None:
                        first_critical_error = ProviderAPIError(error_msg, status_code=status_code, provider=provider_lower, model=model)

        # APIConnectionError
        except APIConnectionError as e:
            app.logger.warning(f"{log_prefix}[Req {attempt_index+1}] API Connection Error for {provider_lower}/{model}: {e}. Request failed.", exc_info=False)
        # Unexpected errors
        except Exception as e:
            error_msg = f"Unexpected error during API call for {provider_lower}/{model}: {type(e).__name__}"
            app.logger.error(f"{log_prefix}[Req {attempt_index+1}] {error_msg} - {e}", exc_info=True)
            with response_lock:
                if first_critical_error is None:
                    first_critical_error = ProviderAPIError(f"{error_msg}. See server logs.", status_code=500, provider=provider_lower, model=model)

    # --- Threading Logic ---
    try:
        for i in range(num_samples):
            with response_lock:
                if first_critical_error is not None:
                    app.logger.warning(f"{log_prefix}Stopping thread launch early due to critical error: {first_critical_error}")
                    break
            thread = threading.Thread(target=send_single_request, args=(i,))
            thread.start()
            threads.append(thread)
            if (i + 1) % batch_size == 0: time.sleep(0.1) # Simple stagger
            elif (i + 1) % 5 == 0: time.sleep(0.03)

        app.logger.debug(f"{log_prefix}Waiting for {len(threads)} launched threads to complete...")
        for i, thread in enumerate(threads):
            thread.join(timeout=90)
            if thread.is_alive(): app.logger.warning(f"{log_prefix}Thread {i+1} timed out.")

        # --- Check for Critical Errors After Joining ---
        with response_lock:
            if first_critical_error is not None:
                app.logger.error(f"{log_prefix}Critical error during batch API calls. Raising ProviderAPIError: {first_critical_error}")
                raise first_critical_error

        app.logger.info(f"{log_prefix}Finished query for {model}. Attempted {len(threads)} samples, successfully collected {len(responses)} responses ({completed_count} reported via callback).")
        if len(responses) < completed_count: # Sanity check
             app.logger.warning(f"{log_prefix}Mismatch between responses collected ({len(responses)}) and callback count ({completed_count}).")
        if completed_count < num_samples and first_critical_error is None:
            app.logger.warning(f"{log_prefix}Collected fewer responses ({completed_count}) than requested ({num_samples}) due to non-critical errors.")

    except ProviderAPIError: raise
    except Exception as e:
        app.logger.error(f"{log_prefix}Critical error during threading/dispatch for {provider_lower}/{model}: {str(e)}", exc_info=True)
        raise ProviderAPIError(f"Threading or dispatch error: {str(e)}", provider=provider_lower, model=model, status_code=500) from e

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
        # Extract words using the regex
        words = re.findall(r'(?:\d+\.\s*)?([A-Za-z]+)(?=\s*\||\s*$)', response)
        # Convert to lowercase and add to list
        all_words.extend(word.lower() for word in words)

    word_frequencies = Counter(all_words)
    app.logger.info(f"Processed {processed_count} valid responses (skipped {skipped_count}). Found {len(word_frequencies)} unique words. Total word occurrences: {len(all_words)}")
    if word_frequencies:
        top_5 = word_frequencies.most_common(5)
        app.logger.debug(f"Top 5 words: {top_5}")

    return word_frequencies

# --- Background Worker Function ---
def run_identification_task(task_id, api_key, provider, model, num_samples, batch_size, temperature):
    """The actual workhorse function that runs in a background thread."""
    log_prefix = f"[Task={task_id}] "
    app.logger.info(f"{log_prefix}Worker thread started for {provider}/{model}.")

    def update_progress(completed, total):
        """Callback function passed to query_llm."""
        with task_lock:
            if task_id in tasks:
                tasks[task_id]['completed_samples'] = completed
                # Update status to processing if it was pending
                if tasks[task_id]['status'] == 'pending':
                    tasks[task_id]['status'] = 'processing'
                app.logger.debug(f"{log_prefix}Progress update: {completed}/{total} samples.")
            else:
                 app.logger.warning(f"{log_prefix}Progress update for non-existent task ID.")


    try:
        # 1. Query LLM with progress reporting
        app.logger.info(f"{log_prefix}Starting LLM query...")
        responses = query_llm(
            api_key, provider, model,
            num_samples, batch_size, temperature,
            progress_callback=update_progress,
            task_id=task_id # Pass task_id for logging context
        )

        # After query finishes, update status if still pending/processing
        with task_lock:
            if task_id in tasks and tasks[task_id]['status'] in ['pending', 'processing']:
                 tasks[task_id]['status'] = 'processing' # Ensure it's marked as processing past query stage
                 tasks[task_id]['completed_samples'] = len(responses) # Final count based on actual returns


        if not responses:
            raise ValueError(f"Failed to collect any responses from {provider}/{model}.")
        app.logger.info(f"{log_prefix}Collected {len(responses)} responses.")

        # 2. Process Responses
        word_frequencies = process_responses(responses)
        if not word_frequencies:
            raise ValueError("No valid words extracted from responses.")
        app.logger.info(f"{log_prefix}Extracted {len(word_frequencies)} unique words.")

        # 3. Prepare Features
        features = prepare_features(word_frequencies)
        if np.sum(features) == 0:
            app.logger.warning(f"{log_prefix}Feature vector is all zeros (no overlap with training words).")

        # 4. Predict
        if classifier is None:
             raise RuntimeError("Classifier model is not loaded.")

        app.logger.info(f"{log_prefix}Making prediction...")
        raw_prediction = classifier.predict(features)[0]

        prediction = raw_prediction
        # Safely convert numpy types if needed
        if isinstance(raw_prediction, (np.integer, np.int64)): prediction = int(raw_prediction)
        elif hasattr(raw_prediction, 'item'): prediction = raw_prediction.item()

        class_labels = []
        if hasattr(classifier, 'classes_'): class_labels = classifier.classes_.tolist()
        else: app.logger.warning(f"{log_prefix}Classifier missing 'classes_' attribute.")

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
                app.logger.warning(f"{log_prefix}Prediction '{prediction}' type {type(prediction)} invalid or out of range.")
        else:
            app.logger.error(f"{log_prefix}Cannot determine model name: class labels unavailable.")

        app.logger.info(f"{log_prefix}Predicted Model='{predicted_model}', Index={predicted_index}")

        confidence = 0.0
        top_predictions = []
        if hasattr(classifier, 'predict_proba') and class_labels and predicted_index != -1:
            try:
                probabilities = classifier.predict_proba(features)[0]
                if len(probabilities) == len(class_labels):
                    sorted_indices = np.argsort(probabilities)[::-1]
                    top_predictions = [
                        {"model": class_labels[i], "probability": float(probabilities[i])}
                        for i in sorted_indices[:5] # Keep top 5
                    ]
                    confidence = float(probabilities[predicted_index])
                    preds_log = [{'m': p['model'], 'p': f"{p['probability']:.4f}"} for p in top_predictions]
                    app.logger.info(f"{log_prefix}Confidence: {confidence:.4f}, Top 5: {preds_log}")
                else:
                    app.logger.error(f"{log_prefix}Probability array length mismatch ({len(probabilities)} vs {len(class_labels)})")
            except Exception as proba_error:
                app.logger.error(f"{log_prefix}Error getting probabilities: {proba_error}", exc_info=False)


        # 5. Prepare Final Result
        status_message = "success"
        final_predicted_model = predicted_model
        if predicted_model == "unknown":
            status_message = "success_unrecognized"
            final_predicted_model = "unrecognized_model"
        elif np.sum(features) == 0:
            status_message = "success_no_overlap"

        result_data = {
            "provider": provider,
            "input_model": model,
            "samples_collected": len(responses),
            "unique_words_extracted": len(word_frequencies),
            "predicted_model": final_predicted_model,
            "confidence": f"{confidence:.2%}",
            "confidence_value": confidence,
            "top_predictions": top_predictions,
            "word_frequencies_top": dict(word_frequencies.most_common(20)), # Keep top 20 words
            "status": status_message, # Add overall status if needed, distinct from task status
            # "processing_time_seconds": calculated later
        }

        # 6. Update Task State to Completed
        with task_lock:
            if task_id in tasks:
                tasks[task_id]['status'] = 'completed'
                tasks[task_id]['result'] = result_data
                tasks[task_id]['end_time'] = time.time()
                duration = tasks[task_id]['end_time'] - tasks[task_id]['start_time']
                tasks[task_id]['result']['processing_time_seconds'] = round(duration, 2)
                app.logger.info(f"{log_prefix}Task completed successfully in {duration:.2f} seconds.")
            else:
                app.logger.error(f"{log_prefix}Task ID disappeared before completion update.")

    except ProviderAPIError as api_err: # Catch critical API errors from query_llm
        error_message = f"API Error ({api_err.status_code}): {api_err.message}"
        app.logger.error(f"{log_prefix}Worker failed due to ProviderAPIError: {error_message}", exc_info=False)
        with task_lock:
            if task_id in tasks:
                tasks[task_id]['status'] = 'error'
                tasks[task_id]['error_message'] = error_message
                tasks[task_id]['end_time'] = time.time()
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        app.logger.error(f"{log_prefix}Worker failed: {error_message}", exc_info=True)
        with task_lock:
            if task_id in tasks:
                tasks[task_id]['status'] = 'error'
                tasks[task_id]['error_message'] = error_message
                tasks[task_id]['end_time'] = time.time()

# --- Flask Routes ---

@app.route('/')
def home():
    app.logger.info("Serving home page")
    return render_template('index.html')

# --- Identify Model Endpoint (MODIFIED to Start Task) ---
@app.route('/api/identify-model', methods=['POST'])
def identify_model_start():
    start_time = time.time()
    app.logger.info("Received request to START /api/identify-model task")
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    num_samples = data.get('num_samples', 100)
    temperature = data.get('temperature', 0.7)
    batch_size = data.get('batch_size', 10)

    # --- Input Validation (same as before) ---
    try: num_samples = min(max(int(num_samples), 10), 1000)
    except: num_samples = 100
    try: temperature = max(0.0, min(float(temperature), 2.0))
    except: temperature = 0.7
    try: batch_size = min(max(int(batch_size), 1), 20)
    except: batch_size = 10

    log_api_key_snippet = f"{api_key[:4]}..." if api_key and len(api_key) > 4 else "Provided" if api_key else "None"
    app.logger.info(f"Identify START request params: Provider={provider}, Model={model}, Samples={num_samples}, Temp={temperature}, Batch={batch_size}, APIKey={log_api_key_snippet}")

    if not api_key or not provider or not model:
        return jsonify({"error": "Missing API key, provider, or model"}), 400
    if classifier is None:
        return jsonify({"error": "Classifier model not loaded."}), 500
    if provider.lower() not in PROVIDER_BASE_URLS:
         app.logger.warning(f"Provider '{provider}' not explicitly listed. Attempting with default OpenAI base URL.")

    # --- Create and Start Task ---
    task_id = str(uuid.uuid4())
    task_info = {
        "task_id": task_id,
        "status": "pending", # Initial status
        "provider": provider,
        "model": model,
        "total_samples": num_samples,
        "completed_samples": 0,
        "result": None,
        "error_message": None,
        "start_time": time.time(),
        "end_time": None
    }

    with task_lock:
        tasks[task_id] = task_info

    app.logger.info(f"[Task={task_id}] Created task. Starting background worker...")

    # Start the background worker thread
    thread = threading.Thread(
        target=run_identification_task,
        args=(task_id, api_key, provider, model, num_samples, batch_size, temperature),
        daemon=True # Allows app to exit even if thread is running (optional)
    )
    thread.start()

    duration = time.time() - start_time
    app.logger.info(f"[Task={task_id}] Identification task accepted and started in background ({duration:.3f}s). Returning task ID.")

    # Return the Task ID immediately
    return jsonify({"task_id": task_id}), 202 # 202 Accepted


# --- Task Status Endpoint (NEW) ---
@app.route('/api/task-status/<string:task_id>', methods=['GET'])
def get_task_status(task_id):
    app.logger.debug(f"Request received for /api/task-status/{task_id}")

    with task_lock:
        task_info = tasks.get(task_id)

    if not task_info:
        app.logger.warning(f"Task status request for unknown task_id: {task_id}")
        return jsonify({"error": "Task not found"}), 404

    status = task_info['status']
    response_data = {"status": status}

    if status == 'pending' or status == 'processing':
        response_data["completed_samples"] = task_info.get('completed_samples', 0)
        response_data["total_samples"] = task_info.get('total_samples', 1) # Avoid division by zero on client
        app.logger.debug(f"Task {task_id} status: {status}, Progress: {response_data['completed_samples']}/{response_data['total_samples']}")
    elif status == 'completed':
        response_data["result"] = task_info.get('result')
        app.logger.debug(f"Task {task_id} status: completed.")
        # Optional: Clean up completed tasks after retrieval?
        # with task_lock:
        #     if task_id in tasks: del tasks[task_id]
    elif status == 'error':
        response_data["message"] = task_info.get('error_message', 'Unknown error')
        app.logger.debug(f"Task {task_id} status: error, Message: {response_data['message']}")
        # Optional: Clean up failed tasks?
        # with task_lock:
        #     if task_id in tasks: del tasks[task_id]

    return jsonify(response_data)


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
    supported_providers = list(PROVIDER_BASE_URLS.keys())
    return jsonify({
        "models": known_models,
        "supported_providers": supported_providers
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
    temperature = data.get('temperature', 0.1)

    try: temperature = max(0.0, min(float(temperature), 1.0))
    except: temperature = 0.1

    log_api_key_snippet = f"{api_key[:4]}..." if api_key and len(api_key) > 4 else "Provided" if api_key else "None"
    app.logger.info(f"Test connection params: Provider={provider}, Model={model}, Temp={temperature}, APIKey={log_api_key_snippet}")

    if not api_key or not provider or not model:
        return jsonify({"error": "Missing API key, provider, or model"}), 400
    if provider.lower() not in PROVIDER_BASE_URLS:
        app.logger.warning(f"Provider '{provider}' not explicitly listed for test. Attempting with default OpenAI base URL.")

    try:
        app.logger.info(f"Attempting single query to test connection to {provider}/{model}")
        # Use num_samples=1, batch_size=1
        responses = query_llm(api_key, provider, model, num_samples=1, temperature=temperature, batch_size=1) # No callback needed here
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
            return jsonify({ "status": "error", "message": f"Failed to get response from '{model}' via '{provider}'. Check credentials, model name, Base URL, and provider status.", "processing_time_seconds": round(duration, 2) }), 500

    except ProviderAPIError as api_err: # Catch specific API errors
         duration = time.time() - start_time
         app.logger.error(f"Test connection ProviderAPIError for {provider}/{model} ({duration:.2f}s): {api_err}", exc_info=False)
         return jsonify({
            "status": "error",
            "message": f"API Error ({api_err.status_code}): {api_err.message}",
            "processing_time_seconds": round(duration, 2)
        }), api_err.status_code or 500
    except Exception as e:
        duration = time.time() - start_time
        app.logger.error(f"Test connection unexpected error for {provider}/{model} ({duration:.2f}s): {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
            "processing_time_seconds": round(duration, 2)
        }), 500


# --- Custom Exception for API Errors ---
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
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.logger.info(f"Starting Flask application. Debug mode: {debug_mode}, Port: {port}")
    # Note: Flask's built-in server is not recommended for production. Use a proper WSGI server (e.g., Gunicorn, uWSGI).
    # Also, the reloader in debug mode might interfere with background threads/in-memory state.
    use_reloader = debug_mode and os.environ.get('WERKZEUG_RUN_MAIN') != 'true' # Avoid reloader issues with threads
    app.run(debug=debug_mode, host='0.0.0.0', port=port, use_reloader=use_reloader) # Disable reloader if debugging threads
