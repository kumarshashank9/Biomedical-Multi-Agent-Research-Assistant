import os

from typing import TypedDict, List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate


class RAGState(TypedDict):
    query: str
    retrieved_documents: List[Document]
    prompt_for_llm: List[BaseMessage]
    llm_answer: str
    retrieved_documents_scores: List[float]
    retrieved_documents_scores_reason: List[str]
    retrieved_documents_correct: List[Document]
    retrieved_documents_ambiguous: List[Document]
    retrieved_documents_incorrect: List[Document]
    retrieved_documents_verdict: str


class ScoreDocs(BaseModel):
    score: float
    reason: str


load_dotenv()
data_path = "../data"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm_model = ChatOpenAI(model='gpt-4o-mini')

vector_store = Chroma(
    collection_name="biomedical_pdf_embeddings",
    persist_directory=os.path.join(data_path, "chroma_db"),
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_type="mmr",
                                      search_kwargs={"k": 5,
                                                     "fetch_k": 20})


def Retrieval(state: RAGState) -> RAGState:
    query = state["query"]
    retrieved_documents = retriever.invoke(query)

    return {"retrieved_documents": retrieved_documents}


def Evaluate_Retrieved_Docs(state: RAGState) -> RAGState:
    query = state["query"]
    retrieved_documents = state["retrieved_documents"]
    grader_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a strict biomedical relevance grader. Your job is to evaluate whether a 
    retrieved document chunk is useful for answering a user's research query.

    Score the document between 0.0 and 1.0 using these guidelines:
    - 0.8 – 1.0 → directly answers the query with specific evidence, findings, or data
    - 0.5 – 0.7 → contains relevant context that partially supports the answer
    - 0.2 – 0.4 → loosely related, mentions relevant concepts but does not address the query
    - 0.0 – 0.1 → completely irrelevant to the query

    Be conservative with high scores. Only award 0.8 and above when the document explicitly 
    and directly addresses the query with concrete evidence. Background information, 
    tangentially related concepts, or general topic mentions should score in the 0.2–0.4 range. 
    When in doubt, score lower rather than higher.

    Provide your score and a brief and short reason explaining why the document received that score."""
        ),
        (
            "human",
            """User Query: {query}

    Retrieved Document:
    {page_content}"""
        )
    ])
    grader_llm = llm_model.with_structured_output(ScoreDocs)
    grader_chain = grader_prompt | grader_llm

    retrieved_documents_scores = []
    retrieved_documents_scores_reason = []
    retrieved_documents_correct = []
    retrieved_documents_ambiguos = []
    retrieved_documents_incorrect = []

    UPPER_THR = 0.7
    LOWER_THR = 0.3

    for doc in retrieved_documents:
        grader_output = grader_chain.invoke({"query": query,
                                             "page_content": doc.page_content})
        if grader_output.score > UPPER_THR:
            retrieved_documents_correct.append(doc)
        elif LOWER_THR <= grader_output.score <= UPPER_THR:
            retrieved_documents_ambiguos.append(doc)
        else:
            retrieved_documents_incorrect.append(doc)

        retrieved_documents_scores.append(grader_output.score)
        retrieved_documents_scores_reason.append(grader_output.reason)

    if len(retrieved_documents_correct) > 0:
        retrieved_documents_verdict = "CORRECT"
    elif len(retrieved_documents_correct) == 0 and len(retrieved_documents_incorrect) == len(retrieved_documents):
        retrieved_documents_verdict = "INCORRECT"
    else:
        retrieved_documents_verdict = "AMBIGUOUS"

    return {"retrieved_documents_scores": retrieved_documents_scores,
            "retrieved_documents_scores_reason": retrieved_documents_scores_reason,
            "retrieved_documents_verdict": retrieved_documents_verdict,
            "retrieved_documents_correct": retrieved_documents_correct,
            "retrieved_documents_ambiguos": retrieved_documents_ambiguos,
            "retrieved_documents_incorrect": retrieved_documents_incorrect}


def Prompt_Generator(state: RAGState) -> RAGState:
    retrieved_documents = state['retrieved_documents_correct']
    systemMessage = """You are a biomedical research assistant with expertise in genomics, 
                        epigenetics, and cancer biology. You answer questions strictly based 
                        on the provided research context. Be precise and scientific in your 
                        language. If the answer is not present in the context, say 
                        "I don't have enough information in the provided documents to answer this." 
                        Do not make up or infer information beyond what is given."""

    context_with_sources = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
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
graph.add_node("Evaluate_Retrieved_Docs", Evaluate_Retrieved_Docs)
graph.add_node("LLM_Generator", LLM_Generator)
graph.add_node("Prompt_Generator", Prompt_Generator)

graph.add_edge(START, "Retrieval")
graph.add_edge("Retrieval", "Evaluate_Retrieved_Docs")
graph.add_edge("Evaluate_Retrieved_Docs", "Prompt_Generator")
graph.add_edge("Prompt_Generator", "LLM_Generator")
graph.add_edge("LLM_Generator", END)

workflow = graph.compile()

initial_state = {
    "query": "Who is the author of paper tiled 'Multimodal analysis of cell-free DNA wholemethylome sequencing for cancer detection and localization'?"}
print(workflow.invoke(initial_state))

png_bytes = workflow.get_graph().draw_mermaid_png()

with open("Biomedical_RAG_workflow.png", "wb") as f:
    f.write(png_bytes)
