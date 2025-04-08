document.addEventListener('DOMContentLoaded', function() {
  const socket = io();
  let clientSid = null;

  socket.on('connect', () => {
    clientSid = socket.id;
    console.log('Socket connected with SID:', clientSid);
  });

  // --- Listen for progress updates from backend ---
  socket.on('progress', (data) => {
    console.log('Progress update received:', data); // More specific log

    // Get elements inside the handler to ensure they are found
    const sampleCountElement = document.getElementById('sample-count');
    const sampleProgressElement = document.getElementById('sample-progress');
    const progressCard = document.getElementById('progress-card');

    // Ensure the progress card is actually visible when updating
    if (progressCard && !progressCard.classList.contains('hidden')) {
        const count = data.current !== undefined ? data.current : 0; // Ensure count is a number
        const total = data.total !== undefined && data.total > 0 ? data.total : 1; // Ensure total is a positive number

        if (sampleCountElement) {
            sampleCountElement.textContent = count; // Update the number text
            console.log(`Updating sample-count text to: ${count}`); // Add specific log for debugging
        } else {
            console.error("Element with ID 'sample-count' not found during progress update!");
        }

        if (sampleProgressElement) {
            const percentage = (count / total) * 100;
            sampleProgressElement.style.width = percentage + '%'; // Update the progress bar width
            console.log(`Updating sample-progress width to: ${percentage}%`); // Add specific log for debugging
        } else {
             console.error("Element with ID 'sample-progress' not found during progress update!");
        }
    } else {
        console.log("Progress update received, but progress card is hidden. Ignoring update.");
    }
  });

  // Load list of known models
  fetch('/api/models')
    .then(response => response.json())
    .then(data => {
      const modelsDiv = document.getElementById('known-models');
      modelsDiv.innerHTML = ''; // Clear previous models if any
      if (data.models && Array.isArray(data.models)) {
          data.models.forEach(model => {
            const modelEl = document.createElement('div');
            modelEl.classList.add('model-tag');
            modelEl.textContent = model;
            modelsDiv.appendChild(modelEl);
          });
      } else {
          console.error('Error: Invalid format for known models received.');
      }
    })
    .catch(error => console.error('Error loading models:', error));

  // Toggle visibility functionality
  const toggleVisibilityBtn = document.getElementById('toggle-visibility');
  const inputSection = document.getElementById('input-section');

  if (toggleVisibilityBtn && inputSection) {
    toggleVisibilityBtn.addEventListener('click', function() {
      const isHidden = inputSection.classList.toggle('hidden');
      this.textContent = isHidden ? 'Show Input Fields' : 'Hide Input Fields';
    });
  }

  // Test connection button
  document.getElementById('test-connection').addEventListener('click', function() {
    const apiKey = document.getElementById('api-key').value;
    const provider = document.getElementById('provider').value;
    const model = document.getElementById('model').value;
    const temperature = parseFloat(document.getElementById('temperature').value);

    if (!apiKey || !provider || !model) {
      showError('Please fill in all fields: API key, provider, and model name.');
      return;
    }

    showCard('testing-card');

    fetch('/api/test-connection', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        api_key: apiKey,
        provider: provider,
        model: model,
        temperature: temperature
      })
    })
      .then(response => response.json())
      .then(data => {
        hideCard('testing-card');
        if (data.status === 'success') {
          alert('Connection successful! Sample response: ' + data.response_preview);
        } else {
          showError(data.message || 'Error connecting to provider');
        }
      })
      .catch(error => {
        hideCard('testing-card');
        console.error('Test Connection Error:', error); // Log detailed error
        showError('Network or server error during test connection. Check console for details.');
      });
  });

  // Identify model button
  document.getElementById('identify-model').addEventListener('click', function() {
    const apiKey = document.getElementById('api-key').value;
    const provider = document.getElementById('provider').value;
    const model = document.getElementById('model').value;
    const numSamples = document.getElementById('num-samples').value;
    const temperature = parseFloat(document.getElementById('temperature').value);

    if (!apiKey || !provider || !model) {
      showError('Please fill in all fields: API key, provider, and model name.');
      return;
    }
    if (!clientSid) {
      showError('Socket not connected yet. Please wait a moment and try again.');
      return;
    }


    // Reset progress display *before* showing the card
    const sampleCountElement = document.getElementById('sample-count');
    const sampleProgressElement = document.getElementById('sample-progress');
    if (sampleCountElement) sampleCountElement.textContent = '0';
    if (sampleProgressElement) sampleProgressElement.style.width = '0%';

    // Show progress card
    showCard('progress-card');


    fetch('/api/identify-model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        api_key: apiKey,
        provider: provider,
        model: model,
        num_samples: parseInt(numSamples),
        temperature: temperature,
        client_sid: clientSid // <-- pass the socket ID
      })
    })
      .then(response => {
        if (!response.ok) {
          // Try to parse error json, otherwise use status text
          return response.json().then(data => {
             throw new Error(data.error || `Server error: ${response.statusText}`);
          }).catch(() => { // Catch if response wasn't JSON
             throw new Error(`Server error: ${response.status} ${response.statusText}`);
          });
        }
        return response.json();
      })
      .then(data => {
        hideCard('progress-card');
        displayResults(data);
      })
      .catch(error => {
        hideCard('progress-card');
        console.error('Identify Model Error:', error); // Log detailed error
        showError(`Error identifying model: ${error.message}. Check console for details.`);
      });
  });

  // Cancel button - Note: This only hides the UI, doesn't stop the backend
  document.getElementById('cancel-identification').addEventListener('click', function() {
    hideCard('progress-card');
    console.log("Identification cancelled by user (UI hidden).");
    // Optionally, send a message to the backend via socket to try and cancel
    // if (clientSid) {
    //   socket.emit('cancel_identification', { sid: clientSid });
    // }
  });

  // Back button from error
  document.getElementById('back-to-form').addEventListener('click', function() {
    hideCard('error-card');
    // Optionally show the main input card again
    // document.getElementById('input-section').classList.remove('hidden');
    // document.getElementById('toggle-visibility').textContent = 'Hide Input Fields';
  });

  // --- Helper Functions ---

  // Store references to card elements to avoid repeated lookups
  const cards = {
      testing: document.getElementById('testing-card'),
      progress: document.getElementById('progress-card'),
      results: document.getElementById('results-card'),
      error: document.getElementById('error-card')
  };

  function showCard(cardKey) {
    // Hide all dynamic cards first
    Object.values(cards).forEach(card => card?.classList.add('hidden'));
    // Show the requested card
    if (cards[cardKey]) {
      cards[cardKey].classList.remove('hidden');
    } else if(document.getElementById(cardKey)) { // Fallback for IDs not in `cards` object
        document.getElementById(cardKey).classList.remove('hidden');
    } else {
        console.error(`Card with key/ID '${cardKey}' not found.`);
    }
  }

  function hideCard(cardKey) {
     if (cards[cardKey]) {
      cards[cardKey].classList.add('hidden');
    } else if(document.getElementById(cardKey)) { // Fallback for IDs not in `cards` object
         document.getElementById(cardKey).classList.add('hidden');
    } else {
        console.error(`Card with key/ID '${cardKey}' not found.`);
    }
  }

  function showError(message) {
    const errorMsgElement = document.getElementById('error-message');
    if(errorMsgElement){
        errorMsgElement.textContent = message;
    }
    showCard('error'); // Use the key 'error'
  }

 function displayResults(data) {
    if (!data) {
        showError("Received invalid data for results display.");
        return;
    }

    try {
        // Fill in basic info
        document.getElementById('input-model').textContent = data.input_model || 'N/A';
        document.getElementById('provider-name').textContent = data.provider || 'N/A';
        document.getElementById('predicted-model').textContent = data.predicted_model || 'N/A';
        document.getElementById('confidence').textContent = data.confidence !== undefined ? `${(data.confidence * 100).toFixed(2)}%` : 'N/A'; // Format confidence

        // Fill in top predictions table
        const predictionsTableBody = document.getElementById('top-predictions')?.querySelector('tbody');
        if (predictionsTableBody) {
            predictionsTableBody.innerHTML = ''; // Clear previous results
            if (data.top_predictions && Array.isArray(data.top_predictions)) {
                data.top_predictions.forEach(prediction => {
                    const row = predictionsTableBody.insertRow();
                    row.insertCell().textContent = prediction.model || 'Unknown Model';
                    row.insertCell().textContent = (prediction.probability * 100).toFixed(2) + '%';
                });
            } else {
                 predictionsTableBody.innerHTML = '<tr><td colspan="2">No prediction data available.</td></tr>';
            }
        } else {
            console.error("Element 'top-predictions' table body not found.");
        }


        // Fill in word frequencies table
        const wordFreqTableBody = document.getElementById('word-frequencies')?.querySelector('tbody');
         if (wordFreqTableBody) {
            wordFreqTableBody.innerHTML = ''; // Clear previous results
            if (data.word_frequencies_top && typeof data.word_frequencies_top === 'object') {
                 const sortedWords = Object.entries(data.word_frequencies_top).sort((a, b) => b[1] - a[1]);
                 const totalWords = sortedWords.reduce((sum, [_, freq]) => sum + freq, 0);

                 if (sortedWords.length === 0) {
                     wordFreqTableBody.innerHTML = '<tr><td colspan="3">No word frequency data available.</td></tr>';
                 } else {
                     // Show top N words (e.g., 20)
                     sortedWords.slice(0, 20).forEach(([word, freq]) => {
                        const row = wordFreqTableBody.insertRow();
                        row.insertCell().textContent = word;
                        row.insertCell().textContent = freq;
                        const percentage = totalWords > 0 ? ((freq / totalWords) * 100).toFixed(2) + '%' : 'N/A';
                        row.insertCell().textContent = percentage;
                    });
                 }
            } else {
                wordFreqTableBody.innerHTML = '<tr><td colspan="3">Word frequency data missing or invalid.</td></tr>';
            }
         } else {
             console.error("Element 'word-frequencies' table body not found.");
         }

      const percentCell = document.createElement('td');
      percentCell.textContent = ((freq / totalWords) * 100).toFixed(2) + '%';
      row.appendChild(percentCell);
      wordFreqTable.appendChild(row);


        // Show the results card
        showCard('results'); // Use the key 'results'

    } catch (e) {
         console.error("Error displaying results:", e);
         showError(`Failed to display results due to an internal error: ${e.message}`);
    }
  }
});
