
import json
from app.OllamaAdapter import OllamaAdapter
from pathlib import Path
import re

llm = OllamaAdapter()


def detect_doc_type_and_metadata(file_path: str, sample_text: str):

    prompt = f"""
    Extract metadata from this university document.

    Return JSON with possible fields:
    - doc_type
    - module_code
    - module_title
    - department
    - assignment
    - deadline
    - year
    

    Only return JSON.

    Document text:
    {sample_text[:2000]}
    """

    response = llm.generate(prompt)

    try:
        metadata = json.loads(response)
    except:
        metadata = {"doc_type": "unknown"}

    metadata["source_file"] = file_path

    return metadata


# def extract_module_metadata(full_text: str):
   

#     module_code = re.search(r"Module Code:\s*(CM\d{4})", full_text)
#     module_title = re.search(r"Title:\s*(.+)", full_text)
#     school = re.search(r"School:\s*(.+)", full_text)

#     return {
#         "doc_type": "module_descriptor",
#         "module_code": module_code.group(1) if module_code else "Unknown",
#         "module_title": module_title.group(1).strip() if module_title else "Unknown",
#         "school": school.group(1).strip() if school else "Unknown",
#     }

