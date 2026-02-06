import logging
import inngest
import inngest.fast_api  # Automatically exposes Inngest endpoints inside FastAPI app
import os
import datetime
import uuid
from fastapi import FastAPI, HTTPException, Header
from inngest.experimental import ai  # Simplifies calling LLMs, managing retries, background AI execution
from dotenv import load_dotenv
# from app.api.routes.ask import router as ask_router

load_dotenv()  # Load environment variables from .env file

inngest_client = inngest.Inngest(   # Initialize Inngest client for handling serverless functions and AI tasks
    app_id="rag_api",               # Unique identifier for Inngest application
    logger = logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)
app = FastAPI(                # creates a web server
    title="LLM-RAG API", 
    version="1.0.0"
    )     

# app.include_router(ask_router, prefix="/api")   # take all the routes defined in ask router and add them to the main app with the prefix /api

@inngest_client.create_function(
    fn_id= "RAG: Ingest Document",  # Unique identifier for the function
    trigger=inngest.TriggerEvent(event="rag/inngest-document")  # Specifies the event that triggers this function
)
async def ingest_document(ctx: inngest.Context):
    return {"message": "Document ingested successfully", "timestamp": datetime.datetime.now().isoformat()}  


@app.get("/health")
def health_check():
    return {"status": "backend is running"} # defines a simple endpoint to check if the backend is running. When you access /health, it will return a JSON response indicating the status of the backend.

inngest.fast_api.serve(app,inngest_client,[ingest_document])  # Bridge Inngest with FastAPI, enabling serverless function execution
