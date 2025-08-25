from dataclasses import dataclass
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from typing import Iterable
from langchain_core.documents import Document
from enum import Enum
from dataclasses import dataclass
from typing import List
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_ollama import ChatOllama
from config import Config
from data_ingestor import ingest_files
from file_loader import File

#Instrução geral para o chatbot. Como será o seu comportamento
SYSTEM_PROMPT = """
Você está conversando com um usuário sobre trechos de seus arquivos.
Tente ser útil e responder às perguntas dele.
Se você não souber a resposta, diga que não sabe e tente fazer perguntas de esclarecimento.
""".strip()

# PROMPT é um placeholder queo LangChain irá preencher dinamicamente antes de enviar para o modelo
PROMPT = """
Aqui estão as informações que você tem sobre os trechos dos arquivos:

<context>
{context}
</context>

Um arquivo pode ter vários trechos.

Por favor, responda à pergunta abaixo:

<question>
{question}
</question>

Answer:
"""
# formato de cada arquivo ou trecho que será injetado no {context}
FILE_TEMPLATE = """
<file>
    <name>{name}</name>
    <content>{content}</content>
</file>
""".strip()

# LangChain que monta prompts antes de enviar para o modelo. 
# Uma estrutura de mensagens no formato de chat multi-turno, ou seja, com papéis de system, user e assistant.
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        #Armazena o histórico. Permite conversas contextuais.
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", PROMPT),
    ]
)


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class ChunkEvent:
    content: str

@dataclass
class SourcesEvent:
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

# Um dicionário que armazena toda a situação da conversa
class State(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer: str

# Pega a string de resultado após o "</think>" do modelo
def _remove_thinking_from_message(message: str) -> str:
    close_tag = "</think>"
    tag_length = len(close_tag)
    return message[message.find(close_tag) + tag_length:].strip()

# Mostra a mensagem de início da conversa
def create_history(welcome_message: Message) -> List[Message]:
    return [welcome_message]

class Chatbot:
    def __init__(self, files: List[File]):
        self.files = files
        self.retriever = ingest_files(files)
        self.llm = ChatOllama(
            model=Config.Model.NAME,
            temperature=Config.Model.TEMPERATURE,
            verbose=False,
            keep_alive=-1,
        )
        self.workflow = self._create_workflow()

    # Chatbot usa isso para transformar cada Document em XML
    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(
            FILE_TEMPLATE.format(name=doc.metadata["source"], content=doc.page_content)
            for doc in docs
        )

    # Usa o retriever para buscar trechos relevantes dos arquivos
    def _retrieve(self, state: State):
        context = self.retriever.invoke(state["question"])
        # Retorna um dicionário com os documentos relevantes no campo context
        return {"context": context}
    
    # Gera a resposta/prompt com o modelo
    def _generate(self, state: State) -> dict:
        messages = PROMPT_TEMPLATE.invoke({
            "question": state["question"],
            "context": self._format_docs(state["context"]),
            "chat_history": state["chat_history"],
        })

        answer = self.llm.invoke(messages)

        return {"answer": answer}
    
    # Cria um StateGraph para gerenciar o fluxo do chatbot:
    # 1° Usuário faz uma pergunta.
    # 2° O workflow executa _retrieve
    # 3° O workflow executa _generate
    def _create_workflow(self) -> CompiledStateGraph:
        graph_builder = StateGraph(State).add_sequence([
            self._retrieve,
            self._generate
        ])
        graph_builder.add_edge(START, "_retrieve")
        return graph_builder.compile()
    

    def _ask_model(
        self, 
        prompt: str, 
        chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:

        # Converte o histórico em mensagens compatíveis com o modelo
        history = [
            AIMessage(m.content) if m.role == Role.ASSISTANT else HumanMessage(m.content)
            for m in chat_history
        ]

        # Prepara o payload da requisição para o workflow
        payload = {
            "question": prompt,
            "chat_history": history
        }

        # Configuração adicional, como ID de thread
        config = {
            "configurable": {
                "thread_id": 42
            }
        }

        # Itera sobre os eventos do fluxo da workflow
        for event_type, event_data in self.workflow.stream(
            payload,
            config=config,
            stream_mode=["updates", "messages"]
        ):
            if event_type == "messages":
                chunk, _ = event_data
                yield ChunkEvent(chunk.content)

            if event_type == "updates":
                if "_retrieve" in event_data:
                    documents = event_data["_retrieve"]["context"]
                    yield SourcesEvent(documents)
                if "_generate" in event_data:
                    answer = event_data["_generate"]["answer"]
                    yield FinalAnswerEvent(answer.content)

    def ask(
            self,
            prompt: str,
            chat_history: List[Message]
        ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
            for event in self._ask_model(prompt, chat_history):
                yield event

                if isinstance(event, FinalAnswerEvent):
                    response = _remove_thinking_from_message("".join(event.content))
                    chat_history.append(Message(role=Role.USER, content=prompt))
                    chat_history.append(Message(role=Role.ASSISTANT, content=response))
