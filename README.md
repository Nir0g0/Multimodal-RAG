## Multimodal PDF RAG with Gemini, LangChain, and LangGraph

This project is a **multimodal Retrieval-Augmented Generation (RAG) system** that lets you ask natural-language questions about one or more PDF reports.  
It combines **text + image understanding** so that charts, tables, and figures inside PDFs are also taken into account when answering questions.

This implementation is inspired by and adapted from the **“Multimodal RAG”** tutorial in the LangChain Open Tutorial series:  
[Multimodal RAG (LangChain Open Tutorial)](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/19-cookbook/06-multimodal/10-geminimultimodalrag).

The code has been reshaped into a **single, terminal-based app** that is easy to run locally.

---

## Key Features

- **Multimodal RAG over PDFs**
  - Extracts text from PDFs in **LLM-friendly markdown** using `pymupdf4llm`.
  - Extracts **figures, charts, and tables as images** using `UpstageDocumentParseLoader`.
  - Generates **factual image descriptions** via Google’s Gemini vision model.
  - Merges text and image descriptions into a unified representation per page.

- **Vector Search over Text + Images**
  - Uses `RecursiveCharacterTextSplitter` to create overlapping text chunks.
  - Embeds chunks with `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`).
  - Stores embeddings in a **Chroma** vector store for fast similarity search.

- **LangGraph-based QA Pipeline**
  - Small but realistic RAG workflow built with `StateGraph` from LangGraph.
  - `retrieve` step pulls the most relevant chunks from the vector store.
  - `generate` step uses `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) to answer questions grounded in the retrieved context.

- **Terminal Q&A Interface**
  - Drop PDFs into the `data/` folder.
  - Run the script.
  - Ask questions such as:  
    - “Which economies are considered AI pioneers?”  
    - “Summarize the key findings of this report.”  
    - “What recommendations are made for AI pioneers?”

---

## Project Structure

- `main.py`  
  End-to-end implementation of the multimodal RAG pipeline and interactive CLI.

- `data/`  
  Folder where you place your PDF documents (for example, `BCG-ai-maturity-matrix-nov-2024.pdf`).

- `.env`  
  Not committed to source control. Stores your API keys and configuration.

---

## How It Works (High-Level Architecture)

### 1. Text Extraction (`extract_markdown`)

- Uses `pymupdf4llm.to_markdown` to convert each PDF page into markdown (`page_chunks=True`).
- Output is a list of page-level dictionaries containing:
  - `text`: markdown text of the page.
  - `metadata.page`: page index.

### 2. Layout + Image Parsing (`load_parsed_docs`)

- Uses `UpstageDocumentParseLoader` from `langchain_upstage`:
  - `split="page"` so each element represents one page.
  - `output_format="markdown"`.
  - `base64_encoding=["figure", "chart", "table"]` so visual elements are stored as base64 images.

### 3. Image Description Generation (`create_image_descriptions`)

- For every base64-encoded image, the code:
  - Calls `ChatGoogleGenerativeAI(model="gemini-2.5-flash")` with a **strict, factual-only prompt**.
  - Asks Gemini to:
    - Output `<---image--->` for decorative images, or
    - Precisely list all textual and numerical content for charts/tables/infographics.
  - Wraps the resulting description into a `Document` with page metadata.

This step is what makes the RAG **truly multimodal**: images are translated into structured, text-like descriptions that can be embedded and retrieved alongside normal text.

### 4. Merge Text and Image Descriptions (`merge_text_and_images`)

- Aggregates:
  - Plain markdown text (from `pymupdf4llm`), and
  - Image-description documents (from Gemini),
  **per page**.
- Produces one `Document` per page with:
  - `page_content`: concatenated text + image descriptions.
  - `metadata.source`: original PDF path.
  - `metadata.page`: page index for traceability.

### 5. Chunking and Vector Store (`build_vector_store`)

- All per-page documents from all PDFs are combined.
- Uses `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` to create overlapping chunks.
- Embeds chunks with `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`.
- Stores embeddings in a **local Chroma** instance via `Chroma.from_documents`.

The resulting `vector_store` is a **global handle** used by the retrieval step.

### 6. RAG QA Pipeline with LangGraph

- Defines a typed `State`:
  - `question: str`
  - `context: List[Document]`
  - `answer: str`

- `retrieve(state: State)`:
  - Calls `vector_store.similarity_search(state["question"])`.
  - Logs a preview of the top-matching chunk.
  - Returns the retrieved documents as `context`.

- `generate(state: State)`:
  - Flattens `context` into a single `docs_content` string.
  - Feeds it into a `ChatPromptTemplate` that instructs the model:
    - Use only the provided context.
    - Say “I don’t know” when necessary.
  - Calls `ChatGoogleGenerativeAI(model="gemini-2.5-flash")` to generate and return the final answer.

- A `StateGraph` wires these steps together into a simple RAG graph:
  - `START -> retrieve -> generate`.

### 7. Interactive CLI (`main`)

The `main()` function:

1. Looks for PDFs in the `data/` directory (`get_pdfs_from_data_folder`).
2. Builds the vector store over all found PDFs (`build_vector_store`).
3. Starts a REPL-like loop:
   - Prompts: `Your question: `
   - Runs the LangGraph pipeline: `graph.invoke({"question": question})`.
   - Prints the generated answer.
   - Exits on `exit`, `quit`, `q`, or empty input.

---

## Setup and Installation

### 1. Environment and Dependencies

**Prerequisites**:

- Python 3.10+ (recommended)
- A virtual environment (e.g. `venv` or `conda`)

Install the main dependencies (example):

```bash
pip install \
  python-dotenv \
  pymupdf4llm \
  langchain-upstage \
  langchain-core \
  langchain-text-splitters \
  langchain-chroma \
  langchain-google-genai \
  langgraph
```

You may also need `chromadb` if it is not pulled transitively:

```bash
pip install chromadb
```

### 2. API Keys and `.env`

Create a `.env` file in the project root (same folder as `main.py`) and provide:

```bash
GOOGLE_API_KEY=your_google_api_key_here
UPSTAGE_API_KEY=your_upstage_api_key_here
```

Additional LangChain / LangSmith variables are optional and only required if you want tracing/monitoring.

### 3. Add PDF Documents

- Create a `data/` folder if it does not already exist.
- Place one or more `.pdf` files into `data/`.  
  For example:
  - `data/BCG-ai-maturity-matrix-nov-2024.pdf`

Every `.pdf` in this folder will be automatically included in the knowledge base.

---

## Running the App

From the project root:

```bash
python main.py
```

You should see:

- A list of detected PDFs.
- Progress messages while text and images are processed.
- A prompt:

```text
You can now ask questions about all indexed documents.
Type 'exit' or press Enter on an empty line to quit.

Your question:
```

Example questions:

- “Please list which countries are represented as AI pioneers.”
- “What are the key findings in this report?”
- “How does the report categorize AI adoption archetypes?”

Type `exit`, `quit`, `q`, or just press Enter on an empty line to terminate.

---
