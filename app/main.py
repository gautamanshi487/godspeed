# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import FileResponse
# from app.git_handler import clone_and_extract_md
# from app.vectorsearch import upsert_documents, query_documents
# from dotenv import load_dotenv
# import openai
# import os

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the app!"}

# @app.get("/favicon.ico")
# async def favicon():
#     # Optionally, replace 'path/to/favicon.ico' with the actual location of your favicon file
#     return FileResponse("path/to/favicon.ico")

# @app.post("/upload_git")
# async def upload_git(repo_url: str, branch: str = "main"):
#     docs = clone_and_extract_md(repo_url, branch)
#     upsert_documents(docs)
#     return {"message": f"{len(docs)} markdown documents upserted"}

# @app.post("/upload_file")
# async def upload_file(file: UploadFile = File(...)):
#     text = (await file.read()).decode("utf-8")
#     upsert_documents([text])
#     return {"message": "Uploaded and upserted successfully."}

# @app.post("/query")
# async def query_rag(query_text: str):
#     relevant_docs = query_documents(query_text)
#     context = "\n\n".join(relevant_docs)

#     prompt = f"Answer the question based on the following docs:\n{context}\n\nQuestion: {query_text}"

#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=prompt,
#         max_tokens=200
#     )
#     return {"answer": response["choices"][0]["text"].strip()}

from fastapi import FastAPI
from app.vectorsearch import upsert_documents, query_documents
from app.git_handler import clone_and_extract_all_files
from app.utils import convert_text_to_embeddings

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Vector search app is running!"}

@app.post("/process-repo/")
def process_repo(repo_url: str, branch: str = "main"):
    docs = clone_and_extract_all_files(repo_url, branch)
    documents_with_embeddings = []
    for doc in docs:
        embedding = convert_text_to_embeddings(doc)
        documents_with_embeddings.append({
            "id": doc,
            "values": embedding,
            "metadata": {"text": doc}
        })
    upsert_documents(documents_with_embeddings)
    return {"status": "Documents processed and upserted successfully."}

@app.post("/query/")
def query_vector(text: str, top_k: int = 5):
    embedding = convert_text_to_embeddings(text)
    results = query_documents(embedding, top_k=top_k)
    return results
#  ADD THIS BLOCK TO START THE SERVER
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
