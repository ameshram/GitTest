# Description: This file contains the code for the Streamlit UI for the project.
# Author: Anup Meshram
# Date of Creation: 2024-02-05

# Importing the required libraries
import re
import os
import torch
import shutil
import pandas as pd
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
# from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings

csv_file = "llm_generated_runbook_description.csv"
model_name = 'sentence-transformers/all-mpnet-base-v2'

# Load the data
loader = CSVLoader(csv_file)
data = loader.load()

# Document Splitters
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# Use the recursive character splitter
recur_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=60,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    is_separator_regex=True,
)

# Perform the splits using the splitter
data_splits = recur_splitter.split_documents(data)

### Using embeddings by Model
model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Define the location to persist data
persist_directory = "chroma/"

# # Check if the directory exists
# if os.path.exists(persist_directory):
#     # Remove old database files if any
#     shutil.rmtree(persist_directory)

# Generate and store embeddings
vectordb = Chroma.from_documents(
    documents=data_splits, embedding=hf_embeddings, persist_directory=persist_directory
)

def generate_recommendation(query):
    
    # Retrieve similar chunks based on relevance. We only retrieve 'k' most similar chunks
    similar_chunks = vectordb.similarity_search_with_relevance_scores(query, k=5, score_threshold=0.20)

    # Initialize lists to hold the extracted data
    retrieved_runbooks = []

    # Extract runbook name and description from the similar chunks
    for chunk in similar_chunks:
        chunk_data = chunk[0]  # Assuming chunk[0] contains the document data
        
        # Extracting the runbook name and description using regex (as per your existing logic)
        runbook_name_match = re.search(r"runbook name: (.*?)\n", chunk_data.page_content)
        runbook_description_match = re.search(r"description:\s*(.*)", chunk_data.page_content)

        # Append to the list as a dictionary
        if runbook_name_match and runbook_description_match:
            retrieved_runbooks.append({
                "runbook_name": runbook_name_match.group(1),
                "runbook_description": runbook_description_match.group(1),
                "relevance_score": chunk[1]  # Assuming chunk[1] contains the relevance score
            })

    # For Streamlit return the dictionary
    return retrieved_runbooks  # or return json_result if you need a JSON string

# Define the structure of your Streamlit app
def main():
    st.title("AI Assistant")

    # Text input for the question
    query = st.text_input("Ask a question:", "")

    # Button to generate answer
    if st.button("Get Answer"):
        if query:  # Ensure there's a question provided
            try:
                # Assuming generate_answer function exists and is imported correctly
                results = generate_recommendation(query)
                st.write(results)  # Display the answer
            except Exception as e:
                st.error(f"Error generating answer: {e}")
        else:
            st.error("Please ask a question.")

if __name__ == "__main__":
    main()