import os
import re
import operator
from typing import TypedDict, List, Annotated
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch


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


class RAGState(TypedDict):
    query: str
    retrieved_documents: List[Document]
    prompt_for_llm: List[BaseMessage]

    retrieved_documents_scores: List[float]
    retrieved_documents_scores_reason: List[str]
    retrieved_documents_correct: List[Document]
    retrieved_documents_ambiguous: List[Document]
    retrieved_documents_incorrect: List[Document]
    retrieved_documents_verdict: str

    relevant_strips: Annotated[List[List[str]], operator.add]

    websearch_query: str
    web_docs: List[Document]

    llm_answer: str


class ScoreDocs(BaseModel):
    score: float
    reason: str


class StripBools(BaseModel):
    keep: bool

class GradeDocs(BaseModel):
    keep: bool


class RewrittenQuery(BaseModel):
    rewritten_query: str
    reasoning: str  # why the query was rewritten this way


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

strip_grader_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a relevance filter. Given a query and a short text strip, 
respond with true if the strip contains any information relevant to the query, 
false if it does not. Be strict — only return true if the strip directly 
relates to the query topic."""
    ),
    (
        "human",
        """Query: {query}

Strip: {strip}"""
    )
])

strip_grader_llm = llm_model.with_structured_output(GradeDocs)
strip_grader_chain = strip_grader_prompt | strip_grader_llm

query_rewriter_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a biomedical search query optimizer. The user's original query 
failed to retrieve relevant documents from the local knowledge base. Your job 
is to rewrite the query into a concise, keyword-rich search query optimized 
for web search engines like PubMed or Google Scholar.

Guidelines:
- Strip conversational language and keep only key concepts
- Include specific biomedical terminology, gene names, or method names if present
- Keep the rewritten query under 15 words
- Do not add information not present in the original query"""
    ),
    (
        "human",
        """Original Query: {query}

Rewritten Search Query:"""
    )
])

query_rewriter_llm = llm_model.with_structured_output(RewrittenQuery)
query_rewriter_chain = query_rewriter_prompt | query_rewriter_llm
tavily = TavilySearch(max_results=5)

web_doc_grader_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a biomedical web search result filter. Given a user query and a 
web search result, decide if the result contains information relevant to answering 
the query.

Return true if the result is relevant, false if it is not.
Be strict — only return true if the result directly relates to the query topic 
with useful information. Generic or tangentially related pages should return false."""
    ),
    (
        "human",
        """Query: {query}

Web Result:
{web_doc}

Is this result relevant?"""
    )
])

web_doc_grader_llm = llm_model.with_structured_output(StripBools)
web_doc_grader_chain = web_doc_grader_prompt | web_doc_grader_llm


def Retrieval(state: RAGState) -> RAGState:
    query = state["query"]
    retrieved_documents = retriever.invoke(query)

    return {"retrieved_documents": retrieved_documents}


def Evaluate_Retrieved_Docs(state: RAGState) -> RAGState:
    query = state["query"]
    retrieved_documents = state["retrieved_documents"]

    retrieved_documents_scores = []
    retrieved_documents_scores_reason = []
    retrieved_documents_correct = []
    retrieved_documents_ambiguous = []
    retrieved_documents_incorrect = []

    UPPER_THR = 0.7
    LOWER_THR = 0.3

    for doc in retrieved_documents:
        grader_output = grader_chain.invoke({"query": query,
                                             "page_content": doc.page_content})
        if grader_output.score > UPPER_THR:
            retrieved_documents_correct.append(doc)
        elif LOWER_THR <= grader_output.score <= UPPER_THR:
            retrieved_documents_ambiguous.append(doc)
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
            "retrieved_documents_ambiguous": retrieved_documents_ambiguous,
            "retrieved_documents_incorrect": retrieved_documents_incorrect}


def Route_after_eval(state: RAGState) -> str:
    if state["retrieved_documents_verdict"] == "CORRECT":
        return "Knowledge_Refinement"
    elif state["retrieved_documents_verdict"] == "INCORRECT":
        return "Web_Search"
    else:
        return "Ambiguous_Node"


def decompose_into_strips(doc: Document) -> List[str]:
    # Split on sentence boundaries
    strips = re.split(r'(?<=[.!?])\s+', doc.page_content)
    # Remove empty strips
    strips = [[s.strip(), doc.metadata] for s in strips if s.strip()]
    return strips


def Knowledge_Refinement(state: RAGState) -> RAGState:
    strips = []
    if state["retrieved_documents_verdict"] == "CORRECT":
        key = "retrieved_documents_correct"
    elif state["retrieved_documents_verdict"] == "AMBIGUOUS":
        key = "retrieved_documents_ambiguous"
    else:
        return {"relevant_strips": []}

    for doc in state[key]:
        strips += decompose_into_strips(doc)

    relevant_strips = []
    for strip in strips:
        strip_content, strip_metadata = strip
        if strip_grader_chain.invoke({"query": state["query"],
                                      "strip": strip_content}).keep:
            relevant_strips.append([strip_content, strip_metadata])

    return {"relevant_strips": relevant_strips}


def Web_Search(state: RAGState) -> RAGState:
    query = state["query"]

    websearch_query = query_rewriter_chain.invoke({"query": query}).rewritten_query

    result = tavily.invoke({"query": websearch_query})["results"]

    web_docs = []
    for r in result or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    relevant_strips = []
    for doc in web_docs:
        if web_doc_grader_chain.invoke({"query": query,
                                        "web_doc": doc.page_content}).keep:
            relevant_strips.append([doc.page_content, doc.metadata])

    return {"web_docs": web_docs,
            "websearch_query": websearch_query,
            "relevant_strips": relevant_strips}


def Ambiguous_Node(state: RAGState) -> RAGState:
    return {}


def Prompt_Generator(state: RAGState) -> RAGState:
    relevant_strips = state['relevant_strips']
    systemMessage = """You are a biomedical research assistant with expertise in genomics, 
                        epigenetics, and cancer biology. You answer questions strictly based 
                        on the provided research context. Be precise and scientific in your 
                        language. If the answer is not present in the context, say 
                        "I don't have enough information in the provided documents to answer this." 
                        Do not make up or infer information beyond what is given."""

    context_with_sources = "\n\n".join([
        f"[Source: {metadata.get('source', 'unknown')}, Page: {metadata.get('page', '?')}, url: {metadata.get('url', '?')}]\n{strip}"
        for strip, metadata in relevant_strips
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
graph.add_node("Knowledge_Refinement", Knowledge_Refinement)
graph.add_node("Web_Search", Web_Search)
graph.add_node("Ambiguous_Node", Ambiguous_Node)
graph.add_node("Prompt_Generator", Prompt_Generator)
graph.add_node("LLM_Generator", LLM_Generator)


graph.add_edge(START, "Retrieval")
graph.add_edge("Retrieval", "Evaluate_Retrieved_Docs")
graph.add_conditional_edges("Evaluate_Retrieved_Docs", Route_after_eval,
{
        "Knowledge_Refinement": "Knowledge_Refinement",
        "Web_Search": "Web_Search",
        "Ambiguous_Node": "Ambiguous_Node"
    }
                            )
graph.add_edge("Knowledge_Refinement", "Prompt_Generator")
graph.add_edge("Web_Search", "Prompt_Generator")
graph.add_edge("Ambiguous_Node", "Knowledge_Refinement")
graph.add_edge("Ambiguous_Node", "Web_Search")
graph.add_edge("Prompt_Generator", "LLM_Generator")
graph.add_edge("LLM_Generator", END)

workflow = graph.compile()

initial_state = {
    "query": "What are some top paper released in 2026 on cfDNA?"}
print(workflow.invoke(initial_state)["llm_answer"])

png_bytes = workflow.get_graph().draw_mermaid_png()

with open("Biomedical_RAG_workflow.png", "wb") as f:
    f.write(png_bytes)
