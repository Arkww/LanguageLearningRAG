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

# Charger les embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Charger l'index FAISS localement
index = faiss.read_index("faiss_index/index.faiss")  # Lire l'index

# Créer un mapping ID pour le docstore
index_to_docstore_id = {i: str(i) for i in range(index.ntotal)}  # Crée un mapping ID

# Charger les documents à partir du fichier pickle
with open("faiss_index/index.pkl", "rb") as f:
    documents = pickle.load(f)
print("Documents chargés avec succès.")

# Créer un docstore factice (FAISS ne stocke pas directement les documents)
docstore = InMemoryDocstore({
    str(i): Document(page_content=str(doc)) for i, doc in enumerate(documents)
})

# Créer le vectorstore FAISS
vectorstore = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,  # Mapping des IDs
    embedding_function=embeddings.embed_query
)

# Charger un modèle pour l'interface de génération de questions/réponses
tokenizer = AutoTokenizer.from_pretrained("neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16")
model = AutoModelForCausalLM.from_pretrained("neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16")
# Créer une pipeline HuggingFace pour la génération
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

# Wrap the pipeline with HuggingFacePipeline
hf_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Créer la chaîne RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=hf_llm, retriever=vectorstore.as_retriever())

# Fonction pour poser des questions à RAG
def ask_rag(question):
    try:
        return qa_chain.run(question)
    except Exception as e:
        return f"Erreur dans la génération de la réponse: {e}"

# Exemple d'utilisation
question = "Quel est le rôle de l'intelligence artificielle dans la santé ?"
print(ask_rag(question))
