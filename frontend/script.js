document.getElementById('submit-btn').addEventListener('click', async () => {
  try {
    const response = await fetch('data.json'); // Fetch local JSON file
    if (!response.ok) {
      throw new Error('Failed to fetch data');
    }
    const data = await response.json();

    // Display accuracy results
    const accuracyResults = document.getElementById('accuracy-results');
    accuracyResults.innerHTML = `
      <h3>Accuracy Across Benchmarks:</h3>
      <ul>
        ${data.benchmarks.map(benchmark => `
          <li><strong>${benchmark.name}:</strong> ${benchmark.accuracy}%</li>
        `).join('')}
      </ul>
    `;

    // Display model classification
    const modelClassification = document.getElementById('model-classification');
    modelClassification.innerHTML = `
      <h3>Model Classification:</h3>
      <p>${data.modelClassification}</p>
    `;
  } catch (error) {
    console.error('Error:', error);
    alert('An error occurred while fetching data.');
  }
});