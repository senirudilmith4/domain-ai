from fastapi import FastAPI, HTTPException, Header
from app.api.routes.ask import router as ask_router

app = FastAPI(                # creates a web server
    title="LLM-RAG API", 
    version="1.0.0"
    )     

app.include_router(ask_router, prefix="/api")   # take all the routes defined in ask router and add them to the main app with the prefix /api

@app.get("/health")
def health_check():
    return {"status": "backend is running"} # defines a simple endpoint to check if the backend is running. When you access /health, it will return a JSON response indicating the status of the backend.