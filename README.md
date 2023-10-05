# Babbler Bot QA Model

Welcome to the Babbler Bot QA Model repository. This sub-repo is responsible for handling the Question Answering (QA) functionality. 

## Architecture
![Logo](https://lh3.googleusercontent.com/JuA_MwHSyOD5WspB9LntncVa8aYdV4AlU3nlvoqVV_7I0nckDzLqbU5L83NF7OL9VcDT6WFc6JbSgMIx3_6u_cUCd86hv5EZDspMRinZfpxOX0WPvErbRy5pYJyJlT2PlSHtaRomrdK64C1rA-0wT3g)

The process flow involves:

1. **Embedding Model**: This initial step creates embeddings of the model and stores them in the vector database.

2. **User Query Processing**: When a user asks a question, it goes through the embedding model. The system then searches the vector database for similar embeddings and retrieves the top 3 relevant documents.

3. **Language Model (LLM)**: These top relevant documents along with the user's question are passed to the Language Model, which generates an appropriate answer.

## Components Used

- **Embedding Model**: Instructor Large
- **Vector Database**: ChromaDB
- **Language Model**: Llama-2
- **Backend Framework**: FastAPI
- **Dependency Management**: Langchain
