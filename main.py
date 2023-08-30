import textwrap
import requests
from bs4 import BeautifulSoup
import difflib
from langchain.document_loaders import GutenbergLoader
import os
import uvicorn

import langchain
from fastapi import FastAPI


from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain import PromptTemplate, ConversationChain, LLMChain

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

    Embeddings_path = "/book-vectordb-chroma"
    Persist_directory = "./book-vectordb-chroma"


# Function to search for a book by name and return the best match URL
def search_book_by_name(book_name):
    base_url = "https://www.gutenberg.org/"
    search_url = (
        base_url
        + "ebooks/search/?query="
        + book_name.replace(" ", "+")
        + "&submit_search=Go%21"
    )

    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the best match link based on similarity ratio
    best_match_ratio = 0
    best_match_url = ""

    for link in soup.find_all("li", class_="booklink"):
        link_title = link.find("span", class_="title").get_text()
        similarity_ratio = difflib.SequenceMatcher(
            None, book_name.lower(), link_title.lower()
        ).ratio()
        if similarity_ratio > best_match_ratio:
            best_match_ratio = similarity_ratio
            best_match_url = base_url + link.find("a").get("href")

    return best_match_url


# Function to get the "Plain Text UTF-8" download link from the book page
def get_plain_text_link(book_url):
    response = requests.get(book_url)
    soup = BeautifulSoup(response.content, "html.parser")

    plain_text_link = ""

    for row in soup.find_all("tr"):
        format_cell = row.find("td", class_="unpadded icon_save")
        if format_cell and "Plain Text UTF-8" in format_cell.get_text():
            plain_text_link = format_cell.find("a").get("href")
            break

    return plain_text_link


# Function to get the content of the "Plain Text UTF-8" link
def get_plain_text_content(plain_text_link):
    response = requests.get(plain_text_link)
    content = response.text
    return content


def select_book(book_name):
    best_match_url = search_book_by_name(book_name)

    if best_match_url:
        book_id = best_match_url.split("/")[-1]  # Extract the book ID
        formatted_url = (
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        )
        print(formatted_url)
        loader = GutenbergLoader(formatted_url)
        book_content = loader.load()
        print("Book content loaded.")
        return book_content
    else:
        print("No matching book found.")
        return None


book_embeddings = None


def create_book_embeddings(book_content):
    global book_embeddings
    print("Creating instructor embeddings...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Configuration.split_chunk_size,
        chunk_overlap=Configuration.split_overlap,
    )
    print("Book content split into chunks.")
    texts = text_splitter.split_documents(book_content)

    print("Creating book embeddings...")
    # try:
    #     book_embeddings = Chroma.load(persist_directory = '.',
    #                            collection_name = 'book')
    # except:
    book_embeddings = Chroma.from_documents(
        documents=texts,
        embedding=instructor_embeddings,
        persist_directory="./embeddings/",
        collection_name="book",
    )
    print("Book embeddings created.")

    book_embeddings.add_documents(documents=texts, embedding=instructor_embeddings)
    book_embeddings.persist()


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


def generate_answer_from_embeddings(query, book_embeddings):
    """
    Retrieve documents from the vector database and then pass them to the language model to generate an answer.

    Args:
        query: The user's question.
        book_embeddings: The embeddings of the book.

    Returns:
        The answer to the question.
    """
    retriever = book_embeddings.as_retriever(
        search_kwargs={"k": Configuration.k, "search_type": "similarity"}
    )
    docs = book_embeddings.similarity_search(query)
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
    model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1},
)

if __name__ == "__main__":
    uvicorn.run("main:app", port=8001)

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=Configuration.embeddings_model_repo, model_kwargs={"device": "cuda"}
)
book_content = None


@app.get("/book")
async def get_book(book_name: str):
    print("Getting book...")
    book_content = select_book(book_name)
    create_book_embeddings(book_content)
    return {"status": "success"}


@app.get("/answer")
async def get_answer(query: str):
    global book_embeddings
    return generate_answer_from_embeddings(query, book_embeddings)
