from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    print("faiss ingestion")

    pdf_file_path = "./knowledge/rag-vs-fine-tuning.pdf"

    loader = PyPDFLoader(file_path=pdf_file_path)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    ### FAISS: it is a local vectorstore which help to query pdf data of any size. It enable you to any vector size to be store in RAM.

    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    # vector_store.save_local("faiss_index_rag_fine_tune")

    # new_vectorstore = FAISS.load_local(
    #     "faiss_index_rag_fine_tune", embeddings
    # )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(OpenAI(), retrieval_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": "Give me the gist of RAG vs Fine Tuning in 3 sentences"})

    print("end", result["answer"])
