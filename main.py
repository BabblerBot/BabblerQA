import textwrap
import requests
from bs4 import BeautifulSoup
import difflib
from langchain.document_loaders import GutenbergLoader
from langchain.document_loaders import WikipediaLoader
import os
import uvicorn
import chromadb

import langchain
from fastapi import FastAPI


from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain import PromptTemplate, ConversationChain, LLMChain
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

from langchain.llms import HuggingFacePipeline
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.chains import RetrievalQA, ConversationalRetrievalChain

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain


class Configuration:
    model_name = "llama2-13b"
    temperature = 0.5
    top_p = 0.95
    repetition_penalty = 1.15

    split_chunk_size = 1000
    split_overlap = 100

    embeddings_model_repo = "hkunlp/instructor-large"

    k = 3

    Persist_directory = "./embeddings/"


def remove_project_gutenberg_sections(text):
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_index = text.find(start_marker)
    start_end_index = text.find("***", start_index + len(start_marker))
    end_index = text.find(end_marker)
    text = text[start_end_index + 3 : end_index]
    return text


def does_book_exist(book_id):
    print("Checking for existing instructor embeddings...")
    client = chromadb.PersistentClient(path=Configuration.Persist_directory)
    try:
        collection = client.get_collection(f"BabblerEmbedding-{book_id}")
        print(
            f"Found collection with {collection.count()} embeddings.\nWill be loaded when needed."
        )
        return True
    except ValueError:
        return False


def select_book(book_id, book_name):
    formatted_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(formatted_url)
    loader = GutenbergLoader(formatted_url)
    docs = WikipediaLoader(query=book_name, load_max_docs=1).load()[0].page_content
    book_content = loader.load()[0].page_content
    filtered_book_content = remove_project_gutenberg_sections(book_content)
    filtered_book_content = docs + filtered_book_content
    print("Book content loaded.")
    return [
        Document(page_content=filtered_book_content, metadata={"source": formatted_url})
    ]


book_embeddings = None


def create_book_embeddings(book_content, book_id: str):
    global book_embeddings
    print("Creating instructor embeddings...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Configuration.split_chunk_size,
        chunk_overlap=Configuration.split_overlap,
    )
    print("Book content split into chunks.")
    texts = text_splitter.split_documents(book_content)

    persistent_client = chromadb.PersistentClient(Configuration.Persist_directory)
    print("Creating book embeddings...")
    book_embeddings = Chroma(
        collection_name=f"BabblerEmbedding-{book_id}",
        embedding_function=instructor_embeddings,
        client=persistent_client,
        persist_directory=Configuration.Persist_directory,
    )

    # try:
    #     book_embeddings = Chroma.load(persist_directory = '.',
    #                            collection_name = 'book')
    # except:
    # book_embeddings = Chroma.from_documents(
    #     documents=texts,
    #     embedding=instructor_embeddings,
    #     persist_directory="./embeddings/",
    #     collection_name="book",
    # )

    book_embeddings.add_documents(documents=texts)
    book_embeddings.persist()
    print("Book embeddings created.")


def wrap_text_preserve_newlines(text, width=200):  # 110
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print(llm_response)
    ans = wrap_text_preserve_newlines(llm_response["result"])
    sources_used = " \n".join(
        [str(source.metadata["source"]) for source in llm_response["source_documents"]]
    )
    ans = ans + "\n\nSources: \n" + sources_used
    return ans


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def generate_answer_from_embeddings(query, book_id: str):
    """
    Retrieve documents from the vector database and then pass them to the language model to generate an answer.

    Args:
        query: The user's question.
        book_id: The id of the book to search.

    Returns:
        The answer to the question.
    """
    # select books embeddings from database.
    book_embeddings = Chroma(
        persist_directory=Configuration.Persist_directory,
        collection_name=f"BabblerEmbedding-{book_id}",
        embedding_function=instructor_embeddings,
    )
    print(f"Loaded book embeddings from {book_embeddings._collection.name}")
    retriever = book_embeddings.as_retriever(
        search_kwargs={"k": Configuration.k, "search_type": "similarity"}
    )
    docs = book_embeddings.similarity_search(query)
    # a = [doc.page_content for doc in docs]
    # for i in a:
    #     print(i)
    # return "lorem ipsum"
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    llm_response = qa_chain(query)
    ans = process_llm_response(llm_response)

    return ans


app = FastAPI()
REPLICATE_API_TOKEN = "r8_KWM7ZPHF27SufFBDWyTQdAHvU07aUHm2aUjQh"
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
llm = Replicate(
    model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={
        "temperature": 0.75,
        "max_length": 500,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
    },
)

if __name__ == "__main__":
    uvicorn.run("main:app", port=8001)

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=Configuration.embeddings_model_repo, model_kwargs={"device": "cuda"}
)
book_content = None


@app.get("/book")
async def get_book(book_id: str, book_name: str):
    has_book = does_book_exist(book_id)
    if has_book:
        return {"status": "success"}

    print("Getting book...")
    book_content = select_book(book_id, book_name)
    create_book_embeddings(book_content, book_id)
    return {"status": "success"}


@app.get("/answer")
async def get_answer(query: str, book_id: str):
    # return "Lorem ipsum"
    has_book = does_book_exist(book_id)
    if not has_book:
        return {"status": "error", "message": "Book not found."}
    return generate_answer_from_embeddings(query, book_id)
