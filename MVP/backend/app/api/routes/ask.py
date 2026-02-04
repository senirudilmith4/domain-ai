from fastapi import APIRouter, HTTPException
from app.schemas.ask import QuestionRequest, AnswerResponse
from app.services.qa_service import answer_question

router = APIRouter(tags=["LLM + RAG"]) # Helps define a group of related endpoints in FastAPI

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest): # async helps improve performance by allowing other tasks to run while waiting for I/O operations
    """
    Endpoint to ask a question and get an answer using LLM + RAG.
    """
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get answer") 