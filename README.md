# 📄 RAG Project

A simple **Retrieval Augmented Generation** app that lets you chat with PDF documents using an LLM.

---

## ⚙️ How It Works

```
PDF uploaded → chunked → embedded → stored
User asks question → retrieve chunks → LLM answers
```

---

## 📂 Structure

| File | Purpose |
|---|---|
| `app.py` | Main application entry point |
| `rag_utility.py` | Core RAG logic — chunking, embedding, retrieval |
| `*.pdf` | Sample documents for testing |
| `requirements.txt` | Dependencies |

---

## 🚀 Run

```bash
pip install -r requirements.txt
python app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![RAG](https://img.shields.io/badge/RAG-PDF%20Q%26A-purple?style=flat)
![LLM](https://img.shields.io/badge/LLM-Groq-orange?style=flat)

---

**👨‍💻 [Ayush Rajak](https://github.com/ayushRajak19)** — Aspiring GenAI Engineer
