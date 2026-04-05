import os
import time
import asyncio
import requests
import inngest
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

@st.cache_resource
def get_inngest_client():
    return inngest.Inngest(app_id="rag_app", is_production=False)


async def send_rag_query_event(question: str, top_k: int):
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]


def _inngest_api_base():
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str):
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s=120):
    start = time.time()
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            if run.get("status") in ("Completed", "Succeeded"):
                return run.get("output", {})
        if time.time() - start > timeout_s:
            raise TimeoutError("Timed out waiting for RAG response")
        time.sleep(0.5)