import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="f3efd778-57a4-40b7-a4ca-9ef6d919ae98")

index_name = "quickstart"

pc.create_index(
    name=index_name,
    dimension=8, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

