# Description: This file contains the code for the Streamlit UI for the project.
# Author: Anup Meshram
# Date of Creation: 2024-02-05

# Importing the required libraries
import pandas as pd
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader



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
                # query_embedding = rec.generate_embeddings(pd.DataFrame({"description": [query]}), model_name)[0].squeeze()
                # results = vectordb.similarity_search_with_relevance_scores(query_embedding)
                results = "TESTING..."

                st.write(results)  # Display the answer
            except Exception as e:
                st.error(f"Error generating answer: {e}")
        else:
            st.error("Please ask a question.")

if __name__ == "__main__":
    main()