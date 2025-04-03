document.addEventListener('DOMContentLoaded', function() {
  // Load list of known models
  fetch('/api/models')
    .then(response => response.json())
    .then(data => {
      const modelsDiv = document.getElementById('known-models');
      data.models.forEach(model => {
        const modelEl = document.createElement('div');
        modelEl.classList.add('model-tag');
        modelEl.textContent = model;
        modelsDiv.appendChild(modelEl);
      });
    })
    .catch(error => console.error('Error loading models:', error));

  // Toggle visibility functionality
  const toggleVisibilityBtn = document.getElementById('toggle-visibility');
  const inputSection = document.getElementById('input-section');

  if (toggleVisibilityBtn && inputSection) {
    toggleVisibilityBtn.addEventListener('click', function() {
      if (inputSection.classList.contains('hidden')) {
        inputSection.classList.remove('hidden');
        this.textContent = 'Hide Input Fields';
      } else {
        inputSection.classList.add('hidden');
        this.textContent = 'Show Input Fields';
      }
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
        showError('Error: ' + error.message);
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

    // Show progress card
    showCard('progress-card');
    document.getElementById('sample-count').textContent = '0';
    document.getElementById('sample-progress').style.width = '0%';

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
        temperature: temperature
      })
    })
      .then(response => {
        if (!response.ok) {
          return response.json().then(data => {
            throw new Error(data.error || 'Error identifying model');
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
        showError('Error: ' + error.message);
      });
  });

  // Cancel button
  document.getElementById('cancel-identification').addEventListener('click', function() {
    hideCard('progress-card');
    // Note: Currently no way to cancel an in-progress request,
    // but we can at least hide the progress UI
  });

  // Back button from error
  document.getElementById('back-to-form').addEventListener('click', function() {
    hideCard('error-card');
  });

  // Helper functions
  function showCard(cardId) {
    // Hide all result cards
    document.querySelectorAll('.card').forEach(card => {
      if (card.id === 'testing-card' || card.id === 'progress-card' ||
          card.id === 'results-card' || card.id === 'error-card') {
        card.classList.add('hidden');
      }
    });
    // Show requested card
    document.getElementById(cardId).classList.remove('hidden');
  }

  function hideCard(cardId) {
    document.getElementById(cardId).classList.add('hidden');
  }

  function showError(message) {
    document.getElementById('error-message').textContent = message;
    showCard('error-card');
  }

  function displayResults(data) {
    // Fill in basic info
    document.getElementById('input-model').textContent = data.input_model;
    document.getElementById('provider-name').textContent = data.provider;
    document.getElementById('predicted-model').textContent = data.predicted_model;
    document.getElementById('confidence').textContent = data.confidence;

    // Fill in top predictions table
    const predictionsTable = document.getElementById('top-predictions').querySelector('tbody');
    predictionsTable.innerHTML = '';

    data.top_predictions.forEach(prediction => {
      const row = document.createElement('tr');

      const modelCell = document.createElement('td');
      modelCell.textContent = prediction.model;
      row.appendChild(modelCell);

      const probCell = document.createElement('td');
      probCell.textContent = (prediction.probability * 100).toFixed(2) + '%';
      row.appendChild(probCell);

      predictionsTable.appendChild(row);
    });

    // Fill in word frequencies table
    const wordFreqTable = document.getElementById('word-frequencies').querySelector('tbody');
    wordFreqTable.innerHTML = '';

    // Sort word frequencies by frequency (descending)
    const sortedWords = Object.entries(data.word_frequencies_top).sort((a, b) => b[1] - a[1]);
    const totalWords = sortedWords.reduce((sum, [_, freq]) => sum + freq, 0);

    // Show top 20 words
    sortedWords.slice(0, 20).forEach(([word, freq]) => {
      const row = document.createElement('tr');

      const wordCell = document.createElement('td');
      wordCell.textContent = word;
      row.appendChild(wordCell);

      const freqCell = document.createElement('td');
      freqCell.textContent = freq;
      row.appendChild(freqCell);

      const percentCell = document.createElement('td');
      percentCell.textContent = ((freq / totalWords) * 100).toFixed(2) + '%';
      row.appendChild(percentCell);

      wordFreqTable.appendChild(row);
    });

    // Show the results card
    showCard('results-card');
  }
});
