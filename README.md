# Agentic RAG Assistant
This project implements an **Agentic AI system** using **Retrieval-Augmented Generation (RAG)**. 

The agent is designed to **interact with a collection of documents**, retrieve relevant information, generate answers, and **self-evaluate its outputs**. Unlike a standard question-answering system, this agent can decide **whether its answer is reliable or requires improvement**.

Key capabilities include:

- **Contextual Retrieval:** The agent retrieves information from documents using both vector similarity (ChromaDB) and keyword search (BM25) to ensure accurate grounding.
- **Answer Generation:** Using a language model, the agent produces answers based on the retrieved context.
- **Self-Evaluation:** Each answer is scored on:
  - **Relevance** – How closely the answer matches the question.
  - **Faithfulness** – How well the answer aligns with the retrieved context.
  - **Clarity** – How understandable the answer is.
- **Autonomous Decision-Making:** Based on evaluation scores, the agent dynamically decides to:
  - Accept the answer as-is.
  - Retrieve additional context and refine the response.
  - Abstain from answering if context is insufficient.

This **reflection-action loop** demonstrates **agentic behavior**, enabling the system to **self-correct, reason, and take meaningful actions** instead of producing static answers.  

The project includes a **Streamlit UI** for interactive testing, making it **lightweight, inspectable, and easy to extend**.

## Quickstart Setup

### 1. Clone repo

```shell
git clone https://github.com/m-felices/ai_academy_capstone_project.git
cd ai_academy_capstone_project
```

### 2. Set your [OpenAI API key](https://platform.openai.com/api-keys)

```shell
export OPENAI_API_KEY="sk_..."
```

(or in `.env.`)

### 3. Install dependencies

```shell
pip install -r requirements.txt
```

### 4. Run the app

```shell
streamlit run app.py
```

### 5. Using Docker
```shell
docker compose up --build
```


### 6. Navigate to [http://localhost:8501](http://localhost:8501).

## Data

 - Place PDFs in data/pdf/
 - Place audio files in data/audio/ (not tracked by git)
 - Folders exist via .gitkeep

## Overview

The Agentic RAG Assistant follows a multi-step reasoning pipeline:

1. User submits a question via the UI
2. Relevant document chunks are retrieved using hybrid search
3. The LLM generates an answer grounded in retrieved context
4. The agent evaluates the quality of the answer
5. Based on the evaluation, the agent decides to:
 - Accept the answer
 - Rewrite it 
 - Return “I don’t know” if context is insufficient

This architecture allows the agent to self-reflect on its outputs and take meaningful actions, rather than responding blindly.

## Main components

1. **Document Loader**  
Loads source documents (PDFs and optional audio transcripts) and converts them into a unified `Document` format for downstream processing.

- PDF loaders (e.g., `PyPDFLoader`)
- Optional audio transcription pipeline
- Stored under `data/`

2. **Text Splitter**  
Splits documents into overlapping chunks to balance semantic coherence and retrieval accuracy.

- Configurable `chunk_size` and `chunk_overlap`

3. **Vector Store (ChromaDB)**  
Stores document embeddings for semantic similarity search.

- Persistent local storage
- Enables fast vector-based retrieval
- Powered by OpenAI or compatible embedding models

4. **Keyword Retriever (BM25)**  
Performs lexical search to complement vector retrieval.

- Effective for exact terms, acronyms, and domain-specific words
- Combined with vector search for hybrid retrieval

5. **Agent Controller**  
Orchestrates the full reasoning loop.

- Combines retrieved context
- Calls the LLM for answer generation
- Invokes evaluation and decides next actions

6. **Evaluator**  
Scores generated answers on:

- **Relevance**
- **Faithfulness**
- **Clarity**

Returns structured scores that the agent uses to decide whether to accept or revise an answer.

7. **Streamlit UI**  
Provides a simple web interface for interaction.

- User input textbox
- Displays final answers
- Automatically resets state between questions to avoid unnecessary API calls
8. **Streamlit UI**.
Provides a simple web interface for interaction.
 - User input textbox
- Displays final answers
- Automatically resets state between questions to avoid unnecessary API calls

## Technologies and Libraries
- Python 3.10+
- Streamlit – user interface
- LangChain – document abstraction and utilities
- ChromaDB – vector database
- BM25 (rank-bm25) – keyword search
- OpenAI API – embeddings and LLM
- Docker & Docker Compose – containerization