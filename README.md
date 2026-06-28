# Voice Agent — Al-Noor Clinic for Nutrition and Aesthetics

A real-time, RAG-powered voice assistant designed for **Al-Noor Clinic for Nutrition and Aesthetics** in Benghazi. The assistant is built using the **LiveKit Agents** framework, powered by **Google Gemini** (`gemini-2.5-flash-native-audio-preview` / `gemini-embedding-001`), and utilizes **ChromaDB** as a vector database to retrieve clinic-specific knowledge (doctors, services, FAQs, and booking rules).

---

## 🏗️ Architecture Overview

The system consists of three main components in the `src/` directory:

1. **`build_rag.py`**: Reads JSON data files from `rag_data/`, generates vector embeddings using the Google GenAI SDK, and stores them in a local ChromaDB collection (`chroma_db/`).
2. **`rag_retriever.py`**: Lazy-loads the ChromaDB client and exposes a `retrieve()` function. This function embeds user queries and queries ChromaDB to retrieve the top matching clinic context.
3. **`agent.py`**: Defines the voice assistant agent using LiveKit. It hooks into the Gemini Realtime API and registers a function tool (`search_clinic_info`) that retrieves context from the RAG store when patients ask about services, schedules, pricing, or doctors.

---

## 🛠️ Prerequisites

Before setting up the project, make sure you have:

* **Python 3.12 or newer** installed on your system.
* **uv** (highly recommended Python package manager) installed.
  * To install `uv`, run:
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    *(or visit the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other platforms).*
* A **LiveKit Cloud** account or local server instance setup.
* A **Google Gemini API Key** (for embedding and LLM features).

---

## 🚀 Setup Instructions

Follow these steps to run the voice agent on your machine:

### 1. Clone & Navigate to the Project
```bash
git clone <repository-url>
cd voice-agent
```

### 2. Configure Environment Variables
Copy the template environment file and fill in your keys:
```bash
copy .env.example .env
```
Open `.env` and fill in the values:
* `GEMINI_API_KEY`: Your Google GenAI API key.
* `LIVEKIT_URL`: Your LiveKit project WebSocket URL (starts with `wss://`).
* `LIVEKIT_API_KEY`: Your LiveKit API key.
* `LIVEKIT_API_SECRET`: Your LiveKit API secret.
* `PYTHONUTF8`: Set to `1` (ensures Arabic text is processed correctly without encoding errors).

### 3. Install Dependencies
Using `uv`, installation is fast and handles the virtual environment automatically:
```bash
uv sync
```
*(If you are not using `uv`, create a virtual environment manually, activate it, and run `pip install -r requirements.txt` or `pip install .`)*

---

## 🗂️ Step-by-Step Running Guide

### Step 1: Build the RAG Database
Before starting the agent, you must parse the clinic data and build the vector database. Run the following command:
```bash
uv run src/build_rag.py
```
This script will:
1. Load clinic details, doctors, services, FAQs, and booking rules from `rag_data/`.
2. Generate text embeddings using Gemini.
3. Create a local ChromaDB collection inside `chroma_db/`.

### Step 2: Start the Voice Agent
Start the LiveKit agent in development mode:
```bash
uv run src/agent.py dev
```
Alternatively, you can run the agent in console mode:
```bash
uv run src/agent.py console
```

Once the agent starts, it will listen for incoming room connections from your LiveKit server/sandbox. You can use the [LiveKit Agent Playground](https://agents-playground.livekit.io/) or the LiveKit console sandbox to test the voice interactions in real-time.

---

## 📂 Project Directory Structure

```text
voice-agent/
├── chroma_db/            # Generated ChromaDB collection folder (git-ignored)
├── rag_data/             # Clinic source data in JSON format
│   ├── booking_rules.json
│   ├── clinic_info.json
│   ├── doctors.json
│   ├── faq.json
│   └── services.json
├── src/
│   ├── __init__.py
│   ├── agent.py          # Main LiveKit voice assistant entrypoint
│   ├── build_rag.py      # ChromaDB builder & embedding script
│   └── rag_retriever.py  # Context retriever for agent tools
├── .env                  # Secrets and keys config (git-ignored)
├── .env.example          # Environment variables template
├── pyproject.toml        # Project metadata and dependencies list
└── README.md             # This documentation file
```
