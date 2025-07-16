from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline  # Correct import
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
import os
import pickle

# Charge the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Charge FAISS index
index = faiss.read_index("faiss_index/index.faiss")  

# Create a mapping ID for the docstore
index_to_docstore_id = {i: str(i) for i in range(index.ntotal)}  

# Charge the documents from the pickle file
with open("faiss_index/index.pkl", "rb") as f:
    documents = pickle.load(f)
print("Documents chargés avec succès.")

docstore = InMemoryDocstore({
    str(i): Document(page_content=str(doc)) for i, doc in enumerate(documents)
})

vectorstore = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,  # Mapping des IDs
    embedding_function=embeddings.embed_query
)

# Model for the HuggingFacePipeline
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

# Wrap the pipeline with HuggingFacePipeline
hf_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Create the RetrivalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=hf_llm, retriever=vectorstore.as_retriever())

# Function to generate an answer using the RAG system
def ask_rag(question):
    try:
        return qa_chain.run(question)
    except Exception as e:
        return f"Error generating the answer: {e}"


