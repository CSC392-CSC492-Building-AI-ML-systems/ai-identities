from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import re
import threading
import time
from collections import Counter

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
        # Go up one level from app.py, then into classifiers
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'classifiers', 'mlp_classifier.pkl')
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        print("Model loaded successfully!")
        return classifier
    except pickle.UnpicklingError as e:
        print(f"Error unpickling model: {e}")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
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

def query_llm(api_key, provider, model, num_samples=100, batch_size=10, temperature=0.7):
    """
    Query the LLM with the earth description prompt multiple times and collect responses.
    Supports OpenAI, Anthropic, Google, and Deep Infra providers.

    Args:
        api_key: API key for the provider
        provider: Provider name (e.g., 'openai', 'anthropic', 'google', 'deepinfra')
        model: Model identifier to query
        num_samples: Number of samples to collect (10-1000)
        batch_size: Number of requests to send in parallel (1-20)
        temperature: Controls randomness (0=deterministic, 2=most random)

    Returns:
        List of responses from the LLM
    """
    global response
    responses = []
    prompt = "Describe the earth using only 10 adjectives. You can only use ten words, each separated by a comma."

    # Validate and adjust parameters
    temperature = max(0.0, min(temperature, 2.0))
    num_samples = max(10, min(num_samples, 1000))
    batch_size = max(1, min(batch_size, 20))

    provider = provider.lower()

    try:
        if provider == 'openai':
            import openai
            openai.api_key = api_key

            def send_response():
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=100
                )
                responses.append(response.choices[0].message.content)
            threads = []
            for _ in range(num_samples):
                thread = threading.Thread(target=send_response)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            print(f"Collected {len(responses)}/{num_samples} responses")

        elif provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)
                batch_responses = []

                for _ in range(current_batch):
                    response = client.messages.create(
                        model=model,
                        max_tokens=100,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    batch_responses.append(response.content[0].text)

                responses.extend(batch_responses)
                print(f"Collected {len(responses)}/{num_samples} responses")

        elif provider == 'google':
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                print(f"Starting to collect {num_samples} samples from {model} with temperature {temperature}...")

                # Consider a smaller batch_size if you still hit limits, e.g., batch_size = 5
                batch_size = 10 # Or adjust as needed based on your quota

                responses = []
                REQUEST_DELAY_SECONDS = 2 # Start with 2 seconds, increase if needed

                for i in range(0, num_samples, batch_size):
                    current_batch_target = min(batch_size, num_samples - i)
                    print(f"\nProcessing batch starting at sample {i} (target size: {current_batch_target})...")
                    batch_responses_in_this_run = [] # Store responses for this specific batch run

                    for sample_index_in_batch in range(current_batch_target):
                        actual_sample_number = i + sample_index_in_batch
                        print(f"  Attempting sample {actual_sample_number + 1}/{num_samples}...")

                        max_retries = 5
                        attempt = 0
                        success = False
                        response_text = None # Store the successful response text

                        while attempt < max_retries and not success:
                            try:
                                generation_config = {
                                    "temperature": temperature,
                                    "max_output_tokens": 100, # Ensure this is appropriate
                                }

                                # Re-create model inside loop if config needs changing per request,
                                # otherwise, can be created once before the inner loop.
                                # For simplicity here, keeping it inside.
                                gemini_model = genai.GenerativeModel(
                                    model_name=model,
                                    generation_config=generation_config
                                )

                                response = gemini_model.generate_content(prompt)

                                # Extract response text
                                if hasattr(response, 'text'):
                                    response_text = response.text
                                elif hasattr(response, 'parts') and response.parts:
                                    response_text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                                else:
                                    # Handle cases where the response might be blocked or empty
                                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                                        print(f"    Sample {actual_sample_number + 1} blocked: {response.prompt_feedback.block_reason}")
                                    else:
                                        print(f"    Sample {actual_sample_number + 1} - Unexpected Google response format or empty response.")
                                    response_text = None # Ensure it's None if blocked or unexpected

                                if response_text is not None:
                                    success = True # Mark as successful to exit retry loop
                                    print(f"    Sample {actual_sample_number + 1} succeeded.")
                                # If response_text is None (blocked/empty), loop continues if attempts remain,
                                # but it won't be added later. Exit retry loop if max attempts reached.

                            except Exception as e:
                                attempt += 1
                                error_msg = str(e)
                                print(f"    Google API error (sample {actual_sample_number + 1}, attempt {attempt}): {error_msg}")

                                # Extract retry delay from error message if available
                                retry_delay = None
                                if "retry_delay" in error_msg:
                                    try:
                                        match = re.search(r'seconds: (\d+)', error_msg)
                                        if match:
                                            retry_delay = int(match.group(1))
                                            # Add a small buffer to the suggested delay
                                            retry_delay = min(retry_delay + 2, 120) # Add buffer, cap max delay
                                    except Exception as parse_err:
                                        print(f"      Could not parse retry_delay: {parse_err}")
                                        retry_delay = None # Fallback to exponential backoff

                                # If no specific delay from API, use exponential backoff
                                if retry_delay is None:
                                    retry_delay = min(2 ** attempt, 60) # Exponential backoff capped at 60s

                                if attempt < max_retries:
                                    print(f"      Retrying sample {actual_sample_number + 1} in {retry_delay} seconds...")
                                    time.sleep(retry_delay)
                                else:
                                    print(f"    Max retries reached for sample {actual_sample_number + 1}. Giving up on this sample.")
                                    break # Exit the retry loop for this sample

                        # --- After the retry loop for a single sample ---
                        if success and response_text is not None:
                            batch_responses_in_this_run.append(response_text)
                        # else: # Optional: Log or handle the failure explicitly if needed
                        #    print(f"    Sample {actual_sample_number + 1} failed permanently.")


                        # *** IMPORTANT: Add a delay AFTER each sample attempt cycle ***
                        # Wait even if successful to avoid hitting RPM limits
                        # Don't sleep if it was the last sample in the batch AND last batch overall
                        is_last_sample_in_batch = (sample_index_in_batch == current_batch_target - 1)
                        is_last_batch = (i + current_batch_target >= num_samples)
                        if not (is_last_sample_in_batch and is_last_batch):
                             # Only print sleep message if actually sleeping
                             if success or attempt < max_retries : # Avoid sleep if last attempt failed and we are exiting
                                 print(f"    --- Waiting {REQUEST_DELAY_SECONDS}s before next sample ---")
                                 time.sleep(REQUEST_DELAY_SECONDS)


                    # --- After processing all samples in the current batch ---
                    if batch_responses_in_this_run:
                        responses.extend(batch_responses_in_this_run)
                    print(f"  Batch finished. Collected {len(responses)}/{num_samples} total responses so far.")

                    # The time.sleep(1) between batches might be less critical now,
                    # but can be kept or adjusted. It ensures a pause even if a batch finishes quickly.
                    # if i + current_batch_target < num_samples:
                    #     time.sleep(1) # Optional pause between batches

                print(f"\nFinished collecting responses. Total successful: {len(responses)}/{num_samples}")


            except ImportError:
                print("Google GenerativeAI not installed. Use: pip install google-generativeai")
                return []
            except Exception as general_err:
                print(f"\nAn unexpected error occurred: {general_err}")
                # Optionally return partial results: return responses
                return [] # Or return empty on any major error

        elif provider == 'deepinfra':
            import requests

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 100
            }

            for i in range(0, num_samples, batch_size):
                current_batch = min(batch_size, num_samples - i)
                batch_responses = []

                for _ in range(current_batch):
                    max_retries = 5
                    base_delay = 1  # starting delay in seconds

                    for attempt in range(max_retries):
                        try:
                            response = requests.post(
                                "https://api.deepinfra.com/v1/openai/chat/completions",
                                headers=headers,
                                json=payload,
                                timeout=30
                            )
                            response.raise_for_status()
                            data = response.json()
                            batch_responses.append(data['choices'][0]['message']['content'])
                            break  # Success - exit retry loop

                        except Exception as e:
                            print(f"DeepInfra API error (attempt {attempt + 1}): {str(e)}")
                            if (hasattr(e, 'response') and e.response.status_code in [429, 503]) or attempt == max_retries - 1:
                                if attempt == max_retries - 1:
                                    print("Max retries reached, giving up.")
                                    break
                                # Calculate delay with exponential backoff
                                delay = base_delay * (2 ** attempt)
                                # Use Retry-After header if available, otherwise use calculated delay
                                if hasattr(e, 'response') and 'Retry-After' in e.response.headers:
                                    delay = float(e.response.headers['Retry-After'])
                                delay = min(delay, 60)  # cap at 60 seconds
                                print(f"Retrying in {delay} seconds...")
                                time.sleep(delay)
                            else:
                                break

                responses.extend(batch_responses)
                print(f"Collected {len(responses)}/{num_samples} responses")

                if i + current_batch < num_samples:
                    time.sleep(1)

        else:
            print(f"Unsupported provider: {provider}")
            return []

    except Exception as e:
        print(f"Critical error in query_llm: {str(e)}")
        return responses if responses else []

    return responses

def process_responses(responses):
    """
    Process the responses from the LLM to extract and count words

    Args:
        responses: List of text responses from the LLM

    Returns:
        Dictionary of {word: frequency}
    """
    all_words = []

    for response in responses:
        # Clean and normalize the response
        words = response.lower().strip()

        # Remove any numbering if present
        words = re.sub(r'^\d+\.?\s*', '', words)
        words = re.sub(r'\s*\d+\.?\s*', ',', words)

        # Split by commas and clean each word
        for word in words.split(','):
            word = word.strip().strip('."\'\n\t')
            if word:
                all_words.append(word)

    # Count word frequencies
    word_frequencies = Counter(all_words)
    return dict(word_frequencies)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/identify-model', methods=['POST'])
def identify_model():
    """
    Endpoint to identify model by querying it with the earth description prompt
    and analyzing word frequencies in responses.

    Expected JSON input:
    {
        "api_key": "your_api_key",
        "provider": "provider_name",
        "model": "model_name",
        "num_samples": 100,   // Optional, default is 100
        "temperature": 0.7
    }
    """
    data = request.json
    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    num_samples = int(data.get('num_samples', 100))
    temperature = float(data.get('temperature', 0.7))

    # Validate temperature
    temperature = max(0.0, min(temperature, 2.0))

    # Limit samples to reasonable range
    num_samples = min(max(num_samples, 10), 1000)

    if not api_key or not provider or not model:
        return jsonify({"error": "Missing API key, provider, or model"}), 400

    if classifier is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    try:
        # Query the LLM and collect responses
        print(f"Starting to collect {num_samples} samples from {provider}/{model} with temperature {temperature}...")
        responses = query_llm(api_key, provider, model, num_samples, temperature=temperature)

        if not responses:
            return jsonify({"error": "Failed to collect responses from the model"}), 500

        print(f"Successfully collected {len(responses)} responses")

        # Process responses to get word frequencies
        word_frequencies = process_responses(responses)

        if not word_frequencies:
            return jsonify({"error": "No valid words extracted from responses"}), 500

        print(f"Extracted {len(word_frequencies)} unique words")

        # Prepare features for prediction
        features = prepare_features(word_frequencies)

        # Make prediction - handle both string and index outputs
        raw_prediction = classifier.predict(features)[0]
        print(f"Raw prediction from classifier: {raw_prediction} (type: {type(raw_prediction)})")

        # Convert prediction to string if it's numpy type
        if isinstance(raw_prediction, (np.integer, np.int64)):
            prediction = int(raw_prediction)
        elif hasattr(raw_prediction, 'item'):
            prediction = raw_prediction.item()
        else:
            prediction = raw_prediction

        # Get list of classifier's known classes
        class_labels = classifier.classes_.tolist() if hasattr(classifier, 'classes_') else []

        # Handle prediction output
        predicted_model = "unknown"
        predicted_index = -1

        # Try to interpret as class label
        if isinstance(prediction, str) and prediction in class_labels:
            predicted_model = prediction
            predicted_index = class_labels.index(prediction)
        else:
            # Try to interpret as index
            try:
                prediction_idx = int(prediction)
                if 0 <= prediction_idx < len(class_labels):
                    predicted_model = class_labels[prediction_idx]
                    predicted_index = prediction_idx
                else:
                    print(f"Prediction index out of range: {prediction_idx}")
            except (ValueError, TypeError):
                print(f"Prediction not recognized as model name or valid index: {prediction}")

        print(f"Final prediction: {predicted_model} (index: {predicted_index})")

        # Get probabilities if available
        confidence = 0.0
        top_predictions = []

        if hasattr(classifier, 'predict_proba'):
            probabilities = classifier.predict_proba(features)[0]
            print(f"Raw probabilities: {probabilities}")

            # Get all predictions sorted by probability (descending)
            sorted_indices = np.argsort(probabilities)[::-1]

            # Extract top 5 models and their probabilities
            top_models = []
            top_probs = []
            for i in sorted_indices[:5]:
                if i < len(class_labels):
                    model_name = class_labels[i]
                    top_models.append(model_name)
                    top_probs.append(float(probabilities[i]))

            # Calculate confidence level (probability of the predicted model)
            if predicted_index != -1 and predicted_index < len(probabilities):
                confidence = probabilities[predicted_index]
                print(f"Confidence for predicted model: {confidence:.2%}")
            else:
                print("Could not calculate confidence - invalid predicted index")

            # Create top predictions list
            top_predictions = [
                {"model": model, "probability": float(prob)}
                for model, prob in zip(top_models, top_probs)
            ]
            print(f"Top predictions: {top_predictions}")

        # Prepare response
        model_info = {
            "provider": provider,
            "input_model": model,
            "samples_collected": len(responses),
            "unique_words": len(word_frequencies),
            "predicted_model": predicted_model if predicted_model != "unknown" else "unrecognized_model",
            "confidence": f"{confidence:.2%}",
            "confidence_value": float(confidence),
            "top_predictions": top_predictions,
            "word_frequencies": word_frequencies,
            "status": "success"
        }

        return jsonify(model_info)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            "error": f"Prediction error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Endpoint to return the list of models the classifier knows about"""
    return jsonify({"models": LIST_OF_MODELS})

@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    """
    Test connection to LLM provider

    Expected JSON input:
    {
        "api_key": "your_api_key",
        "provider": "provider_name",
        "model": "model_name",
        "temperature": 0.7
    }
    """
    data = request.json
    api_key = data.get('api_key')
    provider = data.get('provider')
    model = data.get('model')
    temperature = float(data.get('temperature', 0.7))

    if not api_key or not provider or not model:
        return jsonify({"error": "Missing API key, provider, or model"}), 400

    try:
        # Just query once to test connection
        responses = query_llm(api_key, provider, model, num_samples=1, temperature=temperature)

        if responses:
            return jsonify({
                "status": "success",
                "message": "Successfully connected to provider",
                "response": responses[0]
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Connected but received no response"
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Connection error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
