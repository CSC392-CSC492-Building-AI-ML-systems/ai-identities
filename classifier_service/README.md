# LLM Classifier Service

This service provides LLM identification capabilities using pre-trained models and TF-IDF vectorization.

## Quick Start with Docker (Recommended)

### Option 1: Using Docker Compose (Production)
```bash
# From the root directory
docker-compose up classifier-service -d

# Or start all services
docker-compose up -d
```

### Option 2: Direct Docker
```bash
# Build the image
docker build -t llm-classifier .

# Run the container
docker run -p 8000:8000 -v $(pwd)/resources:/app/resources llm-classifier
```

### Option 3: Direct Python (Development)
```bash
# Install dependencies
pip install -r requirements.txt

# Start the service
python start_service.py
```

## API Usage

The service exposes a `/predict` endpoint:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '["Response 1", "Response 2", "Response 3"]'
```

## Response Format

```json
{
  "prediction": [
    ["model_name", 0.85],
    ["another_model", 0.72],
    ["third_model", 0.65]
  ],
  "all_scores": {
    "model_name": 0.85,
    "another_model": 0.72,
    "third_model": 0.65
  }
}
```

## Configuration

Environment variables:
- `CLASSIFIER_HOST`: Host to bind to (default: 0.0.0.0)
- `CLASSIFIER_PORT`: Port to bind to (default: 8000)
- `CLASSIFIER_RELOAD`: Enable auto-reload (default: false)

## Requirements

- Python 3.8+
- scikit-learn
- numpy
- fastapi
- uvicorn

## Model Files

The service requires these pre-trained model files in the `resources/` directory:
- `vectorizer.pkl`: TF-IDF vectorizer
- `library_averages.pkl`: Pre-computed library averages for known LLMs

## Health Check

The service includes a health check endpoint at `/docs` (FastAPI auto-generated docs).

## Docker Commands

```bash
# View logs
docker-compose logs -f classifier-service

# Restart service
docker-compose restart classifier-service

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```
