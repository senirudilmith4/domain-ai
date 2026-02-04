from pydantic import BaseModel
from typing import List


class QuestionRequest(BaseModel): # request model for asking a question(ask endpoint)
    question: str
    top_k: int = 5

class AnswerResponse(BaseModel): # response model for the answer returned by the ask endpoint
    answer: str
    source_documents: List[Source]

class Source(BaseModel): # model for source documents included in the answer response
    document : str
    page: int