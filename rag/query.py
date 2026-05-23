import os

from typing import TypedDict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage


class RAGState(TypedDict):
    query: str
    retrieved_documents: list[Document]
    prompt_for_llm: list[BaseMessage]
    llm_answer: str


load_dotenv()
data_path = "../data"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm_model = ChatOpenAI()

vector_store = Chroma(
    collection_name="biomedical_pdf_embeddings",
    persist_directory=os.path.join(data_path,"chroma_db"),
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_type="mmr",
                                      search_kwargs={"k": 5,
                                                     "fetch_k": 20})


def Retrieval(state: RAGState) -> RAGState:
    query = state["query"]
    retrieved_documents = retriever.invoke(query)

    return {"retrieved_documents": retrieved_documents}


def Prompt_Generator(state: RAGState) -> RAGState:
    retrieved_documents = state['retrieved_documents']
    systemMessage = """You are a biomedical research assistant with expertise in genomics, 
                        epigenetics, and cancer biology. You answer questions strictly based 
                        on the provided research context. Be precise and scientific in your 
                        language. If the answer is not present in the context, say 
                        "I don't have enough information in the provided documents to answer this." 
                        Do not make up or infer information beyond what is given."""

    context_with_sources = "\n\n".join([
        f"[Source: {doc.metadata}]\n{doc.page_content}"
        for doc in retrieved_documents
    ])

    query = state["query"]
    humanMessage = f"""
                    Context: 
                    {context_with_sources}
                    
                    Question:
                    {query}
                    
                    Answer based only on the context above.When citing, mention 
                    the source document and page number as provided in the context.
    """

    messages = [SystemMessage(content=systemMessage), HumanMessage(content=humanMessage)]

    return {"prompt_for_llm": messages}


def LLM_Generator(state: RAGState) -> RAGState:
    prompt = state["prompt_for_llm"]
    response = llm_model.invoke(prompt).content
    return {"llm_answer": response}


graph = StateGraph(RAGState)

graph.add_node("Retrieval", Retrieval)
graph.add_node("LLM_Generator", LLM_Generator)
graph.add_node("Prompt_Generator", Prompt_Generator)

graph.add_edge(START, "Retrieval")
graph.add_edge("Retrieval", "Prompt_Generator")
graph.add_edge("Prompt_Generator", "LLM_Generator")
graph.add_edge("LLM_Generator", END)

workflow = graph.compile()

initial_state = {"query": "What is Fragment end motif?"}
print(workflow.invoke(initial_state)["llm_answer"])

png_bytes = workflow.get_graph().draw_mermaid_png()

with open("Biomedical_RAG_workflow.png", "wb") as f:
    f.write(png_bytes)


