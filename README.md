# 🚀 Multillm Cost-Optimized Assignment Solution

A unified LLM interface that supports **Groq**, **HuggingFace**, and **local Ollama models** like `llama2`, enabling optimized cost, token usage, and easy experimentation.

---

## ✅ Features

- 🔁 Multi-provider support: `Groq`, `HuggingFace`, `Ollama`
- 💰 Cost & token usage tracking
- 🔐 Secure API handling with `.env`
- ⚙️ Retry logic for failed requests
- 🌐 Local or cloud compatible
- 🧩 Easy to extend with more providers

---

## 🧠 Supported Models

| Provider      | Example Models                                 | Notes                        |
|---------------|------------------------------------------------|------------------------------|
| **Groq**      | `llama3-8b-8192`, `mixtral-8x7b`               | OpenAI-compatible            |
| **HuggingFace** | `meta-llama/Llama-2-7b`, `google/flan-t5-xl` | HuggingFace Inference API    |
| **Ollama**    | `llama2`, `mistral`, `gemma`, `codellama`      | Local LLMs via Ollama        |

---

## ⚙️ Local Setup

### 1. Clone the Repo

```bash
git clone https://github.com/devi-harikaa/Multillm_cost_optimized-assignment-solution.git
cd Multillm_cost_optimized-assignment-solution
2. Create Virtual Environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate    # On Windows
# OR
source venv/bin/activate # On Mac/Linux
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Set Up .env File
Create a .env in the project root:

env
Copy
Edit
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=your_huggingface_token
5. Run the App
bash
Copy
Edit
python app.py
Access it on:

cpp
Copy
Edit
http://127.0.0.1:5000
🔌 Using Local Models via Ollama
1. Install Ollama
Download from https://ollama.com

2. Pull a Model
bash
Copy
Edit
ollama pull llama2
You can replace llama2 with others like mistral, gemma, or codellama.

3. Configure Ollama in config/providers.yaml
yaml
Copy
Edit
llama2:
  type: ollama
  model: llama2
  endpoint: http://localhost:11434
App will now use local LLaMA via Ollama!

📁 Project Structure
bash
Copy
Edit
.
├── app.py                          # Flask backend
├── config/
│   └── providers.yaml              # Model & endpoint config
├── services/                       # Provider classes
├── templates/index.html            # Basic UI
├── utils/                          # Logger & token utils
├── logs/                           # Request logs
├── requirements.txt
└── .env                            # API keys
