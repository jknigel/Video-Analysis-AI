# llm_handler.py

from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods, EmbeddingTypes
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# --- Configuration ---
# It's better practice to load these from environment variables or a config file
URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "skills-network"
LLM_MODEL_ID = "meta-llama/llama-3-8b-instruct"
EMBEDDING_MODEL_ID = EmbeddingTypes.IBM_SLATE_30M_ENG.value

def get_llm_model() -> WatsonxLLM:
    """Initializes and returns the Watsonx LLM."""
    credentials = Credentials(url=URL)
    params = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 1024,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0.7,
    }
    return WatsonxLLM(
        model_id=LLM_MODEL_ID,
        url=credentials.get("url"),
        project_id=PROJECT_ID,
        params=params
    )

def get_embedding_model() -> WatsonxEmbeddings:
    """Initializes and returns the Watsonx Embedding model."""
    credentials = Credentials(url=URL)
    return WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        url=credentials.get("url"),
        project_id=PROJECT_ID
    )

def create_vector_store(text_chunks: list[str], embeddings: WatsonxEmbeddings) -> FAISS:
    """Creates a FAISS vector store from text chunks."""
    return FAISS.from_texts(text_chunks, embeddings)

def create_summarization_chain(llm: WatsonxLLM) -> LLMChain:
    """Creates a LangChain for generating summaries."""
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant. Your task is to provide a concise, single-paragraph summary of the provided YouTube video transcript.
    Focus on the main topics and key points of the video.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = PromptTemplate(input_variables=["transcript"], template=template)
    return LLMChain(llm=llm, prompt=prompt, verbose=True)

def create_qa_chain(llm: WatsonxLLM) -> LLMChain:
    """Creates a LangChain for question-answering."""
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert Q&A assistant. Use the provided context from a video transcript to answer the question accurately.
    If the answer is not available in the context, clearly state that.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context: {context}

    Question: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    return LLMChain(llm=llm, prompt=prompt, verbose=True)