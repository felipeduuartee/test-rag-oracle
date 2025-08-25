import random
from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from chatbot import (
    Chatbot,
    ChunkEvent,
    Message,
    Role,
    SourcesEvent,
    create_history,
)
from file_loader import load_uploaded_file

# Frases engraçadas para exibição enquanto o sistema está processando
LOADING_MESSAGES = [
    "Hold on, I'm wrestling with some digital hamsters... literally.",
    "Loading... please try not to panic, you magnificent disaster.",
    "Just a moment, I'm busy f***ing up the space-time continuum. Oops.",
    "Reticulating splines... and my patience.",
    "Please wait while I deal with the cosmic nonsense of the internet.",
    "Working on your query like a drunk octopus on roller skates.",
    "Hold your horses, I'm busy turning coffee into code.",
    "Give me a sec, I'm interrogating the internet's most indecent secrets.",
    "Loading... because even AI needs a moment to contemplate its meaningless existence.",
    "Processing your request like a lazy rockstar at a rave.",
    "Hang tight, I'm busy tickling the algorithms to make them laugh.",
    "I'm on it faster than you can say 'holy sh*t, that's fast!'",
    "Hold on, while I figure out if your request is a cosmic joke.",
    "Loading... please stand by as I debate the meaning of life with Siri.",
    "Just a moment, I'm unleashing the hounds of data upon your query.",
    "Processing... because apparently, digital miracles take time.",
    "Hold up, I'm busy convincing the bits to behave.",
    "Loading... might as well grab a beer, this could take a while.",
    "Just a sec, I'm wrangling these pixels into some semblance of order.",
    "Be right there — just teaching these binary bits some manners.",
]

WELCOME_MESSAGE = Message(
    role=Role.ASSISTANT,
    content="Olá,👋 Como eu posso ajudá-lo hoje?"
)

# Configuração da página do Streamlit
st.set_page_config(
    page_title="C3NLP RAG",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cabeçalhos principais da aplicação
st.header("C3NLP RAG")
st.subheader("Bem vindo,\nInsira arquivo(s) para iniciarmos um papinho.")

# Função de criação do chatbot com cache
@st.cache_resource(show_spinner=False)                      # para não recriar o chatbot toda vez que a página atualiza (Cache).
def create_chatbot(files: List[UploadedFile]) -> Chatbot:
    processed_files = [load_uploaded_file(file) for file in files]
    return Chatbot(processed_files)

def show_upload_documents() -> List[UploadedFile]:
    holder = st.empty()

    with holder.container():
        uploaded_files = st.file_uploader(
            label="Upload PDF files",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True
        )

    if not uploaded_files:
        st.warning("Por favor carrega um arquivo PDF para continuar!")
        st.stop()

    with st.spinner("Analisando documento(s)..."):
        holder.empty()
        return uploaded_files


########################## Fluxo principal ################################################

# Leitura de arquivos e inicialização do chatbot
uploaded_files = show_upload_documents()
chatbot = create_chatbot(uploaded_files)

# Verifica se uma sessão já foi criada, caso contrário, ele ignora (Por causa da cache)
if "messages" not in st.session_state:
    st.session_state.messages = create_history(WELCOME_MESSAGE)

#Configurando sidebar (barra lateral)
with st.sidebar:
    st.title("Seus arquivos")
    #Nomes dos arquivos upados.
    file_list_text = "\n".join([f"- {file.name}" for file in chatbot.files])
    st.markdown(file_list_text)

# Renderizar mensagens da conversa
for message in st.session_state.messages:
    avatar = "🧑" if message.role == Role.USER else "🤖"
    with st.chat_message(message.role.value, avatar=avatar):
        st.markdown(message.content)

# Entrada do usuário
if prompt := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        full_response = ""
        message_placeholder = st.empty()

        message_placeholder.status(
            random.choice(LOADING_MESSAGES),
            state="running"
        )

        for event in chatbot.ask(prompt, st.session_state.messages):
            if isinstance(event, SourcesEvent):
                for i, doc in enumerate(event.content):
                    with st.expander(f"Source #{i + 1}"):
                        st.markdown(doc.page_content)

            if isinstance(event, ChunkEvent):
                chunk = event.content
                full_response += chunk
                message_placeholder.markdown(full_response)
