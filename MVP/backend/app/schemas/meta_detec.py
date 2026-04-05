import json
import logging
from app.OllamaAdapter import OllamaAdapter

llm = OllamaAdapter()

# Single source of truth for doc_type values.
# Must match across detect_doc_type_and_metadata, extract_filters, and CHUNK_SIZES in load_docs.py.
DOC_TYPES = {
    "module_descriptor",  # module descriptors, syllabi, learning outcomes
    "policy",             # university policies (plagiarism, attendance, etc.)
    "form",               # deferral forms, appeal forms, admin forms
    "exam_paper",         # past papers
    "timetable",          # timetables, schedules
    "unknown",
}


def detect_doc_type_and_metadata(file_path: str, sample_text: str) -> dict:
    prompt = f"""
You are a metadata extractor for a university document system.

Analyse the document text below and return a JSON object with these fields:
- "doc_type"     : one of exactly: module_descriptor | policy | form | exam_paper | timetable | unknown
- "module_code"  : e.g. "CM1601" — omit if not present
- "module_title" : e.g. "Data Structures" — omit if not present
- "department"   : e.g. "Computer Science" — omit if not present
- "year"         : academic year e.g. "2024/25" — omit if not present
- "source"       : set this to exactly: {file_path}

Rules:
- Return ONLY valid JSON, no markdown, no explanation.
- Every response MUST include "doc_type" and "source".

Document text:
{sample_text[:2000]}
"""
    response = llm.generate(prompt)

    try:
        cleaned = response.strip().replace("```json", "").replace("```", "")
        metadata = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logging.warning("Metadata extraction failed for %s — using defaults", file_path)
        metadata = {"doc_type": "unknown"}

    # Normalise doc_type to known values
    if metadata.get("doc_type") not in DOC_TYPES:
        metadata["doc_type"] = "unknown"

    # Always ensure source is set (used for deterministic chunk IDs)
    metadata.setdefault("source", file_path)

    return metadata


def extract_filters(question: str) -> dict | None:
    """
    Extract a ChromaDB `where` filter from a natural language question.
    Returns a dict like {"doc_type": "policy"} or None for unfiltered search.
    """
    prompt = f"""
You are a metadata filter extractor for a university document search system.

Given a student question, return a JSON filter object to narrow the search,
or null if no filter applies.

Available filter field and allowed values:
- "doc_type": module_descriptor | policy | form | exam_paper | timetable

Rules:
- Return ONLY valid JSON or null — no markdown, no explanation.
- Only include "doc_type" if you are confident.

Examples:
Question: "What are the learning outcomes for CM1601?"
Output: {{"doc_type": "module_descriptor"}}

Question: "What topics are covered in Data Structures?"
Output: {{"doc_type": "module_descriptor"}}

Question: "What is the university plagiarism policy?"
Output: {{"doc_type": "policy"}}

Question: "How do I defer my coursework?"
Output: {{"doc_type": "form"}}

Question: "What time does the library open?"
Output: null

Now extract filters for:
{question}
"""
    raw = OllamaAdapter().generate(prompt)

    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned)
        # Validate the extracted doc_type is a known value
        if isinstance(result, dict):
            if result.get("doc_type") not in DOC_TYPES:
                logging.warning(
                    "extract_filters returned unknown doc_type '%s' — ignoring filter",
                    result.get("doc_type"),
                )
                return None
        return result  # dict or None
    except (json.JSONDecodeError, ValueError):
        logging.warning("Filter extraction failed for question: %s — falling back to unfiltered", question)
        return None