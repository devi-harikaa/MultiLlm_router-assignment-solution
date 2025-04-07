# 🧠 LLM Provider Framework – Local + Cloud AI Powered Backend

This project offers a modular backend framework to integrate and switch between multiple LLM providers including **Ollama (local models)**, **Groq (cloud API)**, and **HuggingFace (cloud API)**. It includes automatic fallback between providers, token counting, retry logic, and detailed logs.

---

## 📦 Features

- ✅ Multiple LLM providers: Ollama (local), Groq, HuggingFace
- 🔄 Fallback if one provider fails
- 🔐 API Key support for secure cloud calls
- 📊 Token usage tracking
- 📜 JSON-based prompt sending and response handling
- 🧪 Easily extendable for new providers

---

## 🚀 Quick Start

### 📁 Clone the Repository

```bash
git clone https://github.com/your-username/your-llm-project.git
cd your-llm-project
```

### 🧰 Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Edit or create a `config.yaml` file:

```yaml
providers:
  - name: llama2:latest
    type: ollama
    model: llama2
  - name: groq
    type: groq
    model: llama3-8b-8192
    api_key: YOUR_GROQ_API_KEY
  - name: huggingface
    type: huggingface
    model: google/flan-t5-small
    api_key: YOUR_HUGGINGFACE_API_KEY
```

You can also set API keys via environment variables:

```bash
export GROQ_API_KEY="your-groq-key"
export HF_API_KEY="your-hf-key"
```

---

## ⚙️ Running the Server

```bash
python main.py
```

The app will start on:

```
http://127.0.0.1:5000
```

---

## 🧪 Test the API

Use `curl` or Postman to send a test prompt:

```bash
curl -X POST http://127.0.0.1:5000/generate      -H "Content-Type: application/json"      -d '{"prompt": "Explain the theory of relativity", "max_tokens": 100, "temperature": 0.7}'
```

---

## 🤖 Using Local Models via Ollama (llama2, codellama, etc.)

### 🔹 1. Install Ollama

Download and install from: [https://ollama.com](https://ollama.com)

### 🔹 2. Start Ollama

```bash
ollama serve
```

### 🔹 3. Pull a Model (e.g. llama2)

```bash
ollama pull llama2
```

### 🔹 4. Test Locally (optional)

```bash
ollama run llama2
```

> ℹ️ Ensure Ollama is running on `http://localhost:11434`
> 
> If you see timeouts, confirm:
> - Ollama is running
> - Model is downloaded
> - No firewall is blocking port `11434`

---

## ⚠️ Known Issues & Fixes

### 🧨 Ollama Timeout

```log
Ollama request failed: HTTPConnectionPool(host='localhost', port=11434): Read timed out.
```

**Fix:**

```bash
ollama serve        # Make sure it's running
ollama pull llama2  # Pull the required model
```

---

### 🐛 Groq: `'float' object has no attribute 'get'`

Fix in `groq_provider.py`:

```python
usage = result.get("usage") or {}
```

This avoids errors when API returns null, float, or invalid structure.

---

## 🔄 Provider Fallback Order

The system automatically tries the next provider if the current one fails:

```text
llama2 (local via Ollama) → groq → huggingface
```

Logs will clearly show:
- Which provider is initialized
- Which one is selected
- Errors/fallbacks (if any)

---

## 🗂️ Project Structure

```
.
├── main.py                     # Entry point (Flask API)
├── config.yaml                 # Config for all providers
├── services/
│   ├── provider_manager.py     # Handles fallback and initialization
│   └── providers/
│       ├── llama_provider.py   # Ollama integration
│       ├── groq_provider.py    # Groq API integration
│       └── huggingface_provider.py
├── utils/
│   └── logger.py               # Logging setup
├── requirements.txt
└── README.md
```

---

## ✅ Requirements

- Python 3.9+
- `ollama` installed for local model inference
- API keys for Groq & HuggingFace if cloud generation is needed

---
## OUTPUTS
outputs

(https://github.com/devi-harikaa/MultiLlm_router-assignment-solution/blob/f1c56f7b87c6fe818bcc07db3d8c9beecfe66107/outputs/Screenshot%202025-04-07%20211126.png)
