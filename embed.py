from dotenv import load_dotenv

import os

# load environment variables from .env file to help
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("hello")
print(OPENAI_API_KEY)


from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def create_or_update_index(texts, index_name, embedding):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Created new index: {index_name}")
        vectorstore = PineconeVectorStore.from_documents(
            texts,
            embedding=embedding,
            index_name=index_name
        )
    else:
        print(f"Index {index_name} already exists")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding
        )

    return vectorstore

pdf_path = "dark-energy.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
index_name = "pdf-index"

vectorstore = create_or_update_index(texts, index_name, embedding)