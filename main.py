import spacy
from fastapi import FastAPI

app = FastAPI()

# Load the spaCy model containing pretrained word embeddings
nlp = spacy.load("en_core_web_lg")


@app.get("/")
def read_root():
    return {"message": "Word Embedding API"}


@app.get("/embedding")
def get_embedding(word: str):
    doc = nlp(word)
    embedding = doc.vector.tolist()

    return {
        "word": word,
        "embedding": embedding
    }