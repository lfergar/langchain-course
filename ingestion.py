import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
)  # ← was langchain.document_loaders (dead path)
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")

loader = TextLoader(
    "/Users/leandro/work/LangChain-01/langchain-course-rag/langchain-course/mediumblog1.txt"
)
document = loader.load()


print("Splitting document into chunks...")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

text = text_splitter.split_documents(document)

print(f"Creating embeddings for {len(text)} chunks...")

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

print("Creating Pinecone vector store...")

PineconeVectorStore.from_documents(
    text, embeddings, index_name=os.environ.get("PINECONE_INDEX_NAME")
)

print("Ingestion complete!")
