import ast
import re

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_course_transcripts(path: str):
    loader = TextLoader(path)
    return loader.load()


def text_splitter(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def extract_dict_from_text(text: str):
    match = re.search(r"```(?:json|python)?\s*(.*?)\s*```", text, re.DOTALL)

    if match:
        extracted_string = match.group(1).strip()
    else:
        extracted_string = text.strip()

    if extracted_string.startswith("{{") and extracted_string.endswith("}}"):
        extracted_string = extracted_string[1:-1]
    elif extracted_string.startswith("{") and extracted_string.endswith("}"):
        pass

    try:
        final_dict = ast.literal_eval(extracted_string)
    except Exception as e:
        print(f"Error: {e}")
        return None
    return final_dict
