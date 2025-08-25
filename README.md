# 🧠 C3NLP

**C3NLP** é um sistema de **Geração de Respostas Contextuais** utilizando **RAG (Retrieval-Augmented Generation)**, construído com:

- **[Streamlit](https://streamlit.io/)** → Interface web interativa para exibição e upload de arquivos.  
- **[LangChain](https://www.langchain.com/)** → Orquestração de LLMs, memória de conversas e fluxo de RAG.  
- **[Ollama](https://ollama.com/)** → Execução de LLMs localmente, como `LLaMA 3` e `DeepSeek`.  

---

## 📚 O que cada tecnologia faz

- **LangChain**:  
  Framework para criar pipelines de LLMs com suporte a RAG, embeddings, agentes e memória de conversas.

- **Ollama**:  
  Plataforma para executar modelos de LLM localmente, gerenciando downloads e execução com aceleração via GPU.

- **Streamlit**:  
  Framework Python para criar aplicações web rápidas e interativas, perfeito para dashboards e protótipos de IA.

---

## 🤖 Modelos Utilizados

Atualmente, o projeto suporta e utiliza os seguintes modelos:

- **LLMs (via Ollama)**:
  - `llama3`
  - `deepseek-r1:14b`  

- **Embeddings e Rerankers** (para RAG):
  - `BAAI/bge-small-en-v1.5` → Geração de embeddings  
  - `ms-marco-MiniLM-L-12-v2` → Re-ranqueamento semântico  

---

## ⚙️ Pré-processamento de Documentos

1. **Upload do arquivo** (`.pdf`, `.txt`, `.md`)  
2. **Extração de conteúdo**:
   - PDF → Lido com `pypdfium2`  
   - TXT/MD → Lido e decodificado em UTF-8  
3. **Chunking**:
   - **Tamanho:** `2048` tokens  
   - **Sobreposição:** `128` tokens  
4. **Geração de Embeddings**  
5. **Indexação para RAG** com recuperação BM25 + semântica  

---

## 📦 Requisitos

- **Python 3.13.**    
- **[Ollama](https://ollama.com/)** instalado e rodando localmente  
- **Modelos LLM baixados localmente**, por exemplo:
  ```bash
  ollama pull llama3
  ollama pull deepseek-r1:14b
  ```

---

## 🚀 Fluxo de Execução do Chatbot

```mermaid
flowchart TD
    A[Upload de Arquivo] --> B[load_uploaded_file]
    B --> C[Arquivo carregado]
    C --> D[Usuário faz pergunta]
    D --> E[ask]
    E --> F[_ask_model]
    F --> G{workflow.stream}

    G -->|SourcesEvent| H[Documentos recuperados]
    G -->|ChunkEvent| I[Streaming de resposta]
    G -->|FinalAnswerEvent| J[Resposta final]

    J --> K[Atualiza histórico e retorna]



---

## 🖥️ Como Rodar o Projeto

### 1️⃣ Clone o repositório
```bash
git clone git@github.com:Guilherme-Eduardo/C3NLP.git
cd C3NLP
```

### 2️⃣ Crie um ambiente virtual e instale as dependências
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3️⃣ Execute a aplicação
```bash
streamlit run C3NLP/app.py
```

Acesse no navegador: **http://localhost:8501**

---

## 🐳 Rodando com Docker

...