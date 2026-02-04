from app.rag.retriever import retrieve_context
import ollama   

def answer_question(question: str, top_k: int):
    # 1. Retrieve relevant documents (RAG)
    context_docs = retrieve_context(question, top_k)

    # 2. Build context text
    context_text = "\n\n".join(
        [doc["content"] for doc in context_docs]
    )

    # 3. Construct prompt
    system_prompt = (
        "You are a university domain-specific assistant. "
        "Answer only using the provided context."
    )

    full_prompt = f"""
    Context:
    {context_text}

    Question:
    {question}
    """

    # 4. Call LLM
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )

    answer = response["message"]["content"]

    # 5. Format sources
    sources = [
        {
            "document": doc["source"],
            "page": doc["page"]
        }
        for doc in context_docs
    ]

    return {
        "answer": answer,
        "sources": sources
    }
