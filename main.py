from dotenv import load_dotenv
import os
import base64
from collections import defaultdict
from typing_extensions import List, TypedDict

import pymupdf4llm
from langchain_upstage import UpstageDocumentParseLoader
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph


# Load API keys and other configuration from .env for authentication.
load_dotenv(override=True)


def extract_markdown(file_path: str):
    """
    Extract text from a PDF as markdown, split by page.

    This uses pymupdf4llm so that the result is already
    in an LLM‑friendly markdown format.
    """
    md_text = pymupdf4llm.to_markdown(
        doc=file_path,
        page_chunks=True,
        show_progress=True,
    )
    return md_text


def load_parsed_docs(file_path: str):
    """
    Load parsed pages (with image metadata) using Upstage.

    Compared to plain text extraction this also detects figures,
    charts and tables and stores them as base64 so we can send
    them to the vision model later.
    """
    loader = UpstageDocumentParseLoader(
        file_path,
        split="page",
        output_format="markdown",
        base64_encoding=["figure", "chart", "table"],
    )
    docs = loader.load_and_split()
    return docs


def create_image_descriptions(docs):
    """
    For each detected image on each page, call Gemini Vision to
    obtain a strictly factual description (no interpretation).

    The returned list contains one `Document` per image with
    `page` metadata so it can later be merged back into the
    corresponding page text.
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    new_documents = []

    for doc in docs:
        print(f"Processing images on page {doc.metadata.get('page', 'unknown')}")
        if "base64_encodings" in doc.metadata and len(doc.metadata["base64_encodings"]) > 0:
            for img_base64 in doc.metadata["base64_encodings"]:
                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": """
Describe only the factual content visible in the image:

1. If decorative/non-informational: output '<---image--->'

2. For content images:
- General Images: List visible objects, text, and measurable attributes
- Charts/Infographics: State all numerical values and labels present
- Tables: Convert to markdown table format with exact data

Rules:
* Include only directly observable information
* Use original numbers and text without modification
* Avoid any interpretation or analysis
* Preserve all labels and measurements exactly as shown
""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )

                response = model.invoke([message])
                print(f"Image description created for page {doc.metadata.get('page', 'unknown')}")

                new_doc = Document(
                    page_content=response.content,
                    metadata={"page": f"{doc.metadata.get('page', 'unknown')}"},
                )
                new_documents.append(new_doc)

    return new_documents


def merge_text_and_images(md_text, image_description_docs, source_path: str):
    """
    Merge plain page text and image descriptions into a single
    `Document` per page for one PDF.

    The resulting documents keep `source` (PDF path) and `page`
    in their metadata so we can always see where a chunk came from.
    """
    page_contents = defaultdict(list)
    page_metadata = {}

    for text_item in md_text:
        page = int(text_item["metadata"]["page"])
        page_contents[page].append(text_item["text"])
        if page not in page_metadata:
            page_metadata[page] = {
                "source": source_path,
                "page": page,
            }

    for img_doc in image_description_docs:
        page = int(img_doc.metadata["page"])
        page_contents[page].append(img_doc.page_content)

    merged_docs = []
    for page in sorted(page_contents.keys()):
        full_content = "\n\n".join(page_contents[page])
        doc = Document(page_content=full_content, metadata=page_metadata[page])
        merged_docs.append(doc)

    return merged_docs


def build_vector_store(file_paths):
    """
    Build a single Chroma vector store over one or more PDFs.

    Each PDF is processed separately into per‑page documents,
    then all pages across all PDFs are concatenated and chunked
    before being embedded and stored.
    """
    all_documents = []

    for file_path in file_paths:
        print(f"\nLoading and parsing document: {file_path}")
        md_text = extract_markdown(file_path)
        docs = load_parsed_docs(file_path)

        print("Creating image descriptions (if any)...")
        image_description_docs = create_image_descriptions(docs)

        print("Merging text and image descriptions...")
        merged_documents = merge_text_and_images(md_text, image_description_docs, file_path)
        all_documents.extend(merged_documents)

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_documents)

    print("Building vector store over all PDFs...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store_local = Chroma.from_documents(documents=all_splits, embedding=embeddings)

    print("Vector store ready.\n")
    return vector_store_local


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate(
    [
        (
            "human",
            """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Question: {question} 
Context: {context} 
Answer:
""",
        )
    ]
)


class State(TypedDict):
    """
    Shared state passed through the small LangGraph pipeline.

    - `question`: user input from the terminal
    - `context`: list of retrieved `Document` chunks
    - `answer`: final answer generated by the LLM
    """

    question: str
    context: List[Document]
    answer: str


# Global handle used by the `retrieve` step; initialised once
# in `main()` after all PDFs have been processed.
vector_store = None


def retrieve(state: State):
    """
    Retrieve the most relevant chunks for the current question
    from the shared Chroma vector store.
    """
    print(f"\nSEARCHING DOCUMENTS...\n{'='*20}")
    retrieved_docs = vector_store.similarity_search(state["question"])
    if retrieved_docs:
        print(f"Top match preview:\n{retrieved_docs[0].page_content[:200]}...")
    print(f"{'='*20}")
    return {"context": retrieved_docs}


def generate(state: State):
    """
    Turn the retrieved chunks into a single answer.

    The prompt is simple and instructs the model to stay grounded
    in the provided context and admit when it does not know.
    """
    print(f"\nGENERATING ANSWER...\n{'='*20}")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


def get_pdfs_from_data_folder(data_folder: str):
    """
    Find all PDFs under the given `data_folder`.

    This keeps things simple: every `.pdf` file in the directory
    is automatically added to the knowledge base.
    """
    pdf_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in '{data_folder}'.")
        print("Add one or more PDF files to this folder and run the script again.")

    else:
        print(f"Found {len(pdf_files)} PDF files in '{data_folder}':")
        for path in pdf_files:
            print(f"  - {os.path.basename(path)}")

    return pdf_files


def main():
    """
    Entry point for the terminal application.

    1. Collect all PDFs from `data/`
    2. Build a vector store over their contents
    3. Start an interactive Q&A loop in the terminal
    """
    data_folder = "data"
    if not os.path.isdir(data_folder):
        print(f"Data folder '{data_folder}' does not exist.")
        print("Create the folder, add PDF files, and run the script again.")
        return

    pdf_files = get_pdfs_from_data_folder(data_folder)
    if not pdf_files:
        return

    global vector_store
    vector_store = build_vector_store(pdf_files)

    print("You can now ask questions about all indexed documents.")
    print("Type 'exit' or press Enter on an empty line to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if not question or question.lower() in {"exit", "quit", "q"}:
            print("Exiting. Goodbye.")
            break

        response = graph.invoke({"question": question})
        print("\nAnswer:")
        print(response["answer"])
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()