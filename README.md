# Jarvis Knowledge Assistant

A retrieval-augmented chat application that combines a locally hosted LLaMA 2 7B (Q4\_K\_M) model with a Pinecone vector index built from four detailed computer-science study guides (OOP, Computer Architecture, Operating Systems, and SDLC). A lightweight HTML interface lets users converse with “Jarvis” while the backend performs retrieval over Pinecone and produces grounded responses.
- Built by Aditya Dwivedi (24MCAA01) — [LinkedIn](https://www.linkedin.com/in/aditya-dwivedi-622776131/) · [GitHub](https://github.com/Adityadgithub)

## Highlights
- Local inference with `llama-cpp-python` (CPU by default; GPU optional).
- Retrieval-Augmented Generation via Pinecone + sentence-transformer embeddings.
- FastAPI backend with streaming-ready `/chat` endpoint.
- Minimal HTML/JS frontend for quick experimentation.


## Repository Structure
- `backend/`
  - `app.py` – FastAPI server (embeddings, Pinecone retrieval, LLaMA runner).
  - `ingest.py` – Creates Pinecone index from the curated study guides.
  - `config.py` – Paths, model parameters, Pinecone settings.
  - `requirements.txt` – Python dependencies.
- `data/` – Source `.txt` knowledge files.
- `frontend/index.html` – Static chat UI.
- `llm_model/` – Expected location for `llama-2-7b-chat.Q4_K_M.gguf` (not tracked).

## Prerequisites
- Python 3.10+
- Pip (newest version recommended)
- Pinecone account (Starter/workspace with available quota)
- Internet access (first run) to download embeddings and reach Pinecone
- LLaMA 2 7B GGUF weights (see below)

### Model Weights
The 3 GB `llama-2-7b-chat.Q4_K_M.gguf` file is **not** included and should not be committed to GitHub. Instead:
1. Download the quantized model (e.g. from [TheBloke’s Hugging Face repo](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)).
2. Place it at `llm_model/llama-2-7b-chat.Q4_K_M.gguf`.
3. Keep `llm_model/` in `.gitignore` to avoid uploading large binaries.

If you want to share the project, add a note in the README (or separate docs) describing how to obtain the weights.

## Setup
```powershell
cd C:\MYData\Llm_project
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r backend\requirements.txt
```

### Environment Variables
Create a `.env` (not tracked) in the project root:
```dotenv
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-east1-gcp      # or the environment shown in the Pinecone console
PINECONE_INDEX_NAME=jarvis-knowledge-index
PINECONE_NAMESPACE=default
```
The ingestion script parses `PINECONE_ENVIRONMENT` to derive the serverless cloud/region (e.g. `us-east1-gcp` → cloud=`gcp`, region=`us-east1`). Override those defaults by setting `PINECONE_ENVIRONMENT` accordingly.

### Build / Refresh the Pinecone Index
```powershell
cd C:\MYData\Llm_project\backend
python ingest.py
```
Use `python ingest.py --force` to recreate the index after updating files in `data/`.

## Running the Application
```powershell
cd C:\MYData\Llm_project\backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Startup logs should include “Application startup complete” once LLaMA loads and Pinecone connects successfully.

### Frontend
Open `frontend/index.html` in a browser (e.g. `file:///C:/MYData/Llm_project/frontend/index.html`). It sends requests to `http://localhost:8000/chat`.

## Working with GitHub
- **Commit the codebase** (`backend/`, `frontend/`, `data/`, README, etc.).
- **Exclude large artifacts**: add `llm_model/` and `.env` to `.gitignore`.
- **Document model acquisition** so collaborators can reproduce the setup without committing the weights.

Example `.gitignore` entries:
```
.venv/
__pycache__/
.env
llm_model/
```

## Tips
- Adjust `LLM_MAX_NEW_TOKENS`, `LLM_TEMPERATURE`, or `n_gpu_layers` inside `backend/config.py` / `app.py` to tune responses or enable GPU offload.
- Rerun `python ingest.py` whenever you change the `.txt` files in `data/`.
- Keep Pinecone costs in mind; delete unused indexes from the console after testing.

With the `.env` populated and the model weights downloaded locally, you can ingest, run the server, and interact with Jarvis end-to-end. When sharing the project, push the code and instructions—never the proprietary or large binary weights. 