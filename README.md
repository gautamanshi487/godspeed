# godspeed

# GS Documentation Embedder + RAG Agent (Godspeed Framework)

This project is a Retrieval-Augmented Generation (RAG) solution built using the Godspeed framework. It allows users to ingest GitHub documentation or local files into a vector database and query them using natural language powered by LLMs.

 Bonus: Built using the Godspeed Framework as specified in the assignment.

---

# Vector Search App

This is a FastAPI application that provides a vector search functionality. It allows users to upload and process GitHub repositories, and then query the processed documents using a natural language interface.


## Project Structure

```
├── app/               # Contains API code and logic
├── repo_clone/        # Temporary folder for cloned GitHub docs
├── .env               # Environment variables (e.g., OpenAI key)
├── .gitignore
├── requirements.txt   # Python dependencies
├── run.sh             # Startup script
├── gs_rag_env/        # Virtual environment folder
```

---

## Features

* Process GitHub repositories and extract all files
* Convert text to embeddings using a BERT model
* Upsert documents with embeddings into a Pinecone vector search database
* Query the vector search database using a natural language interface
* Return top-k relevant documents based on the query

## Endpoints

### 1. Root Endpoint

* `GET /`: Returns a welcome message indicating that the app is running.

### 2. Process Repository Endpoint

* `POST /process-repo/`: Processes a GitHub repository and extracts all files.
	+ Parameters:
		- `repo_url`: The URL of the GitHub repository to process.
		- `branch`: The branch of the repository to process (default: "main").
	+ Returns: A success message indicating that the documents have been processed and upserted.

### 3. Query Endpoint

* `POST /query/`: Queries the vector search database using a natural language interface.
	+ Parameters:
		- `text`: The query text to search for.
		- `top_k`: The number of top-k relevant documents to return (default: 5).
	+ Returns: A list of relevant documents based on the query.

## Dependencies

* FastAPI
* Pinecone
* Transformers (for BERT model)
* PyTorch

## Environment Variables

* `PINECONE_API_KEY`: The API key for the Pinecone service.
* `PINECONE_ENVIRONMENT`: The environment for the Pinecone service (default: "gcp-starter").
* `PINECONE_INDEX_NAME`: The name of the Pinecone index (default: "my-index").

## Running the App

To run the app, simply execute the following command:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
