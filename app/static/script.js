document.addEventListener("DOMContentLoaded", function () {
    const submitBtn = document.getElementById("submit-btn");
    const apiKeyInput = document.getElementById("api-key");
    const providerSelect = document.getElementById("provider");
    const accuracyResults = document.getElementById("accuracy-results");
    const modelClassification = document.getElementById("model-classification");
    const toggleVisibilityBtn = document.getElementById('toggle-visibility');
    const inputSection = document.getElementById('input-section');

    // Toggle visibility function (modified for initial visibility)
    toggleVisibilityBtn.addEventListener('click', function() {
      if (inputSection.classList.contains('hidden')) {
        inputSection.classList.remove('hidden');
        this.textContent = 'Hide Input Fields'; // Change button text
      } else {
        inputSection.classList.add('hidden');
        this.textContent = 'Show Input Fields'; // Change button text
      }
    });

    submitBtn.addEventListener("click", function () {
        const apiKey = apiKeyInput.value.trim();
        const provider = providerSelect.value;

        if (!apiKey) {
            alert("Please enter your API key.");
            return;
        }

        fetch("/api/identify-model", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ api_key: apiKey, provider: provider })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                accuracyResults.innerHTML = `<p style='color: red;'>Error: ${data.error}</p>`;
            } else {
                accuracyResults.innerHTML = `<p>Model: ${data.model}</p>`;
                modelClassification.innerHTML = `<p>Confidence: ${data.confidence}</p>`;
            }
        })
        .catch(error => {
            accuracyResults.innerHTML = `<p style='color: red;'>An error occurred.</p>`;
            console.error("Error:", error);
        });
    });
});
