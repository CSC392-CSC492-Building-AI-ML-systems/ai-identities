from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import run_inference


HOST = "0.0.0.0"
PORT = 8000

server = FastAPI()

@server.post("/identify")
async def identify_llm_req(responses: list[str] = Body()):
    try:
        result = run_inference.identify_llm(responses)
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500)


if __name__ == "__main__":
    uvicorn.run("prediction_server:server", host=HOST, port=PORT)
