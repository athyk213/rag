# Ensure your VertexAI credentials are configured
import os

import vertexai
from google.cloud import aiplatform
from google.oauth2 import service_account
from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings


def init_gemini_llm(keyfile_path: str):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keyfile_path
    aiplatform.init(project="cs391-project")
    vertexai.init(project="cs391-project")
    return init_chat_model(
        "gemini-2.0-flash-001", model_provider="google_vertexai"
    )


def init_embeddings(keyfile_path: str):
    credentials = service_account.Credentials.from_service_account_file(
        keyfile_path
    )
    return VertexAIEmbeddings(
        model="text-embedding-004", credentials=credentials
    )


# --- OpenAI Functions ---


def init_openai_llm(api_key: str, model_name: str = "gpt-4o-mini"):
    """Initializes the OpenAI LLM."""
    os.environ["OPENAI_API_KEY"] = api_key
    # Example uses gpt-4o-mini, adjust model name as needed
    # gpt-4o-mini, o1-mini, o1-preview
    return init_chat_model(model_name, model_provider="openai")


# Uncoomnet below for testing
# keyfile_path = "/home/mc76728/repo/Coargus/vrag/cs391-project-11f0f788cfea.json"
# llm = init_llm(keyfile_path)
# embeddings = init_embeddings(keyfile_path)

# print(llm.invoke("Hello, world!").content)
# print(np.mean(embeddings.embed_query("Hello, world!"), axis=0))
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # --- OpenAI Functions ---
llm = init_openai_llm(
    model_name="o1-preview",
    api_key="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",
)
print(llm.invoke("Hello, world!").content)
