import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def get_wikitionary_defs(word, lang="en"):
    url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return format_definition(data)  
    return "Definition not found."

def format_definition(definition):
    """Convertit le JSON de Wiktionary en une chaîne de caractères lisible"""
    if isinstance(definition, dict):
        text = []
        for lang, defs in definition.items():
            for d in defs:
                part_of_speech = d.get("partOfSpeech", "unknown")
                meaning = d.get("text", "No definition available")
                text.append(f"[{lang}] ({part_of_speech}): {meaning}")
        return " | ".join(text)
    return str(definition) 


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

data = [
    get_wikitionary_defs("hello", "en"),
    get_wikitionary_defs("bonjour", "fr"),
    get_wikitionary_defs("你好", "zh"),
    get_wikitionary_defs("hallo", "de"),
    "Bonjour est le mot français pour hello.",
    "Hello est le mot anglais pour bonjour.",
    "你好 est le mot chinois pour hello.",
    "Hallo est le mot allemand pour hello.",
    "Bonjour est un mot français.",
    "Hello est un mot anglais.",
    "你好 est un mot chinois.",
    "Hallo est un mot allemand.",
    "Bonjour est un mot qui signifie hello.",
    "Hello est un mot qui signifie bonjour.",
    "Bonjour signifie hello en anglais.",
    "Le kanji 你 signifie 'tu' en chinois.",
    "La déclinaison du mot 'livre' en allemand est différente de celle en français."
]

embeddings = model.encode(data)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype=np.float32))

faiss.write_index(index, "faiss_index/index.faiss")

with open("faiss_index/index.pkl", "wb") as f:
    pickle.dump(data, f)