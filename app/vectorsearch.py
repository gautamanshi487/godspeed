# import os
# from uuid import uuid4
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# import numpy as np
# from transformers import BertTokenizer, BertModel
# import torch

# # Load environment variables from a .env file
# load_dotenv()

# # Initialize Pinecone
# api_key = os.getenv("PINECONE_API_KEY")  # Ensure this is set in your environment
# if not api_key:
#     raise ValueError("PINECONE_API_KEY is not set. Please set it in your environment or .env file.")

# environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")  # Default or use your environment
# index_name = os.getenv("PINECONE_INDEX_NAME", "my-index")

# # Initialize Pinecone client
# pc = Pinecone(api_key=api_key)

# # Create index if it doesn't exist
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=1536,  # Adjust based on your embedding size (BERT typically produces 768 or 1536)
#         metric="cosine",  # You can also use 'euclidean', 'dotproduct', etc.
#         spec=ServerlessSpec(
#             cloud="aws", region="us-west-2"  # Or your correct region
#         )
#     )

# # Get reference to the index
# index = pc.Index(index_name)

# # Convert text to embeddings using HuggingFace's BERT
# def text_to_embeddings(text_list):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
    
#     embeddings = []
#     for text in text_list:
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         outputs = model(**inputs)
#         embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Taking mean of all tokens
#         embeddings.append(embedding)
#     return np.vstack(embeddings)

# # Function to upsert documents (embeddings) to Pinecone
# def upsert_documents(documents):
#     """
#     Upserts a list of documents with embeddings to Pinecone.
#     Each document is a dict: {"id": str, "values": list[float], "metadata": dict}
#     """
#     vectors = [
#         {
#             "id": doc.get("id", str(uuid4())),
#             "values": doc["values"],
#             "metadata": doc.get("metadata", {})
#         }
#         for doc in documents
#     ]
#     index.upsert(vectors)

# # Function to query documents
# def query_documents(embedding, top_k=5, include_metadata=True):
#     """
#     Queries the index with a given embedding.
#     Returns the top_k most similar documents.
#     """
#     result = index.query(
#         vector=embedding,
#         top_k=top_k,
#         include_metadata=include_metadata
#     )
#     return result["matches"]
import os
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")  # Ensure this is set in your environment
if not api_key:
    raise ValueError("PINECONE_API_KEY is not set. Please set it in your environment or .env file.")

environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")  # Default or use your environment
index_name = os.getenv("PINECONE_INDEX_NAME", "my-index")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust based on your embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", region="us-west-2"  # Or your correct region
        )
    )

# Get reference to the index
index = pc.Index(index_name)

# Function to upsert documents (assumes embeddings are already computed)
def upsert_documents(documents):
    """
    Upserts a list of documents with embeddings to Pinecone.
    Each document is a dict: {"id": str, "values": list[float], "metadata": dict}
    """
    vectors = [
        {
            "id": doc.get("id", str(uuid4())),
            "values": doc["values"],
            "metadata": doc.get("metadata", {})
        }
        for doc in documents
    ]
    index.upsert(vectors)

# Function to query documents
def query_documents(embedding, top_k=5, include_metadata=True):
    """
    Queries the index with a given embedding.
    Returns the top_k most similar documents.
    """
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=include_metadata
    )
    return result["matches"]
