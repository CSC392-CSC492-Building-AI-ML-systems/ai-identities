from fastapi import FastAPI, Body, HTTPException
from run_inference import identify_llm

app = FastAPI(title="LLM Classifier Service")

@app.post("/predict")
async def predict(responses: list[str] = Body(..., description="List of responses from the unknown LLM")):
    """
    Endpoint to identify an LLM based on responses.
    """
    try:
        result = identify_llm(responses)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")