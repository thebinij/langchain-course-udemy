from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()
## text-embedding-ada-002 (OpenAI Embeddings)

if __name__ == '__main__':
    print("Start...")
    loader = TextLoader("./knowledge/mediumblog1.txt")
    document = loader.load()

    print("Splitting...")

    ## Rule of thumb for
    ## chunk_size should be small enough so it could fit in the context window and
    ## it should be big enough it can provide value and semantic meaning.
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)

    ## another rule is Garbage in Garbage Out.
    ## Even though the llm may able so have very large context size, sending
    ## unrelavent data will to LLM, firstly it'll be costly and second it's proven
    ## we get worse result.

    ## A chunk_overlap is useful when we want context between chunks.

    texts = text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")

    print("ingesting...")

    PineconeVectorStore.from_documents(texts, embedding=OpenAIEmbeddings(), index_name=os.environ['INDEX_NAME'])

    print("finish")