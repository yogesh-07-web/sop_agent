import re
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.35  # semantic similarity (much better)

def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)


def answer_question(question, retrieved_chunks):
    sentences = []

    for chunk in retrieved_chunks:
        sentences.extend(split_into_sentences(chunk))

    if not sentences:
        return "Answer is not available in the uploaded PDFs."

    question_embedding = model.encode(question, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(question_embedding, sentence_embeddings)[0]
    best_score = float(scores.max())
    best_index = int(scores.argmax())

    if best_score < SIMILARITY_THRESHOLD:
        return "Answer is not available in the uploaded PDFs."

    return sentences[best_index]
