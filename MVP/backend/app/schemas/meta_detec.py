
import json
from app.OllamaAdapter import OllamaAdapter
from pathlib import Path


llm = OllamaAdapter()

def extract_module_code(chunk_text: str) -> str:
    """
    Extracts the module code (e.g., CM2601) from a text chunk using LLM.
    Returns a string or 'Unknown' if not found.
    """
    prompt = f"""
    You are a helpful assistant. Extract the module code from the following text.
    The module code format is always like CMXXXX (letters + 4 digits).
    Respond only with JSON in this format: {{"module_code": "<code>"}}.
    
    Text:
    {chunk_text}
    """

    try:
        response = llm.generate(prompt)
        data = json.loads(response)
        return data.get("module_code", "Unknown")
    except Exception as e:
        print("Error extracting module code:", e)
        return "Unknown"


def extract_module_title(chunk_text: str) -> str:
    """
    Extracts the module title from a text chunk using LLM.
    Returns a string or 'Unknown' if not found.
    """
    prompt = f"""
    You are a helpful assistant. Extract the module title from the following text.
    Respond only with JSON in this format: {{"module_title": "<title>"}}.
    
    Text:
    {chunk_text}
    """

    try:
        response = llm.generate(prompt)
        data = json.loads(response)
        return data.get("module_title", "Unknown")
    except Exception as e:
        print("Error extracting module title:", e)
        return "Unknown"

def extract_school(chunk_text: str) -> str:
    """
    Extracts the school/department name from a text chunk using LLM.
    Returns a string or 'Unknown' if not found.
    """
    prompt = f"""
    You are a helpful assistant. Extract the school or department name from the following text.
    Respond only with JSON in this format: {{"school": "<school_name>"}}.
    
    Text:
    {chunk_text}
    """
    try:
        response = llm.generate(prompt)
        data = json.loads(response)
        return data.get("school", "Unknown")
    except Exception as e:
        print("Error extracting school:", e)
        return "Unknown"


def extract_policy_name(chunk_text: str) -> str:
    """
    Extracts the policy/regulation name from a text chunk using LLM.
    Returns a string or 'Unknown' if not found.
    """
    prompt = f"""
    You are a helpful assistant. Extract the policy or regulation name from the following text.
    Respond only with JSON in this format: {{"policy_name": "<policy_name>"}}.
    
    Text:
    {chunk_text}
    """
    try:
        response = llm.generate(prompt)
        data = json.loads(response)
        return data.get("policy_name", "Unknown")
    except Exception as e:
        print("Error extracting policy name:", e)
        return "Unknown"


def extract_course_name(chunk_text: str) -> str:
    """
    Extracts the course name from a text chunk (e.g., BSc AI & Data Science) using LLM.
    Returns a string or 'Unknown' if not found.
    """
    prompt = f"""
    You are a helpful assistant. Extract the course name from the following text.
    Respond only with JSON in this format: {{"course": "<course_name>"}}.
    
    Text:
    {chunk_text}
    """
    try:
        response = llm.generate(prompt)
        data = json.loads(response)
        return data.get("course", "Unknown")
    except Exception as e:
        print("Error extracting course name:", e)
        return "Unknown"


def extract_year(chunk_text: str) -> str:
    """
    Extracts the year/stage from a text chunk (e.g., 1, 2, 2024, 2025) using LLM.
    Returns a string or 'Unknown' if not found.
    """
    prompt = f"""
    You are a helpful assistant. Extract the year or academic year from the following text.
    Respond only with JSON in this format: {{"year": "<year>"}}.
    
    Text:
    {chunk_text}
    """
    try:
        response = llm.generate(prompt)
        data = json.loads(response)
        return data.get("year", "Unknown")
    except Exception as e:
        print("Error extracting year:", e)
        return "Unknown"

def detect_doc_type_and_metadata(pdf_path: str, chunk: str, page_num: int):
    pdf_name = Path(pdf_path).name
    metadata = {"source": pdf_name, "page": page_num}

    # Detect module descriptor
    if "CM" in chunk and "Module" in chunk:
        metadata.update({
            "doc_type": "module_descriptor",
            "module_code": extract_module_code(chunk),
            "module_title": extract_module_title(chunk),
            "school": extract_school(chunk)
        })
    elif "Policy" in chunk or "Regulation" in chunk:
        metadata.update({
            "doc_type": "policy",
            "policy_name": extract_policy_name(chunk)
        })
    elif "Timetable" in chunk:
        metadata.update({
            "doc_type": "timetable",
            "course": extract_course_name(chunk),
            "year": extract_year(chunk)
        })
    else:
        metadata.update({"doc_type": "generic"})
    
    return metadata



