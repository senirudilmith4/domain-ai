from fastapi import FastAPI
import ollama

app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

@app.post("/generate/")
def generate_text(prompt: str):
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return {"response": response["message"]["content"]}