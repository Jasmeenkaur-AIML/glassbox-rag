import chromadb
import requests

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("notes")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def retrieve(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0], results["distances"][0]


def build_prompt(question, chunks):
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"[Chunk {i+1}]\n{chunk.strip()}\n\n"

    return f"""You are a study assistant answering questions using ONLY the context below.

Rules:
- Answer in 3-5 sentences maximum. Be concise, not exhaustive.
- Only use information from the context. Do not use outside knowledge.
- After your answer, on a new line, write "Source: Chunk X" citing which chunk number(s) you used.
- If the context does not contain enough information to answer, say "I don't have enough information in the notes to answer this confidently." Do not guess.

Context:
{context_block}

Question: {question}

Answer:"""


def generate_answer(prompt):
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    response.raise_for_status()
    return response.json()["response"]


def build_verification_prompt(answer, chunks):
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"[Chunk {i+1}]\n{chunk.strip()}\n\n"

    return f"""You are a strict fact-checker. Below is some CONTEXT and an ANSWER that claims to be based on it.

Your job: decide if the ANSWER is fully supported by the CONTEXT, with no invented or outside information.

Respond in EXACTLY this format, nothing else:
VERDICT: <SUPPORTED or NOT SUPPORTED>
REASON: <one short sentence explaining why>

CONTEXT:
{context_block}

ANSWER:
{answer.strip()}

Your response:"""


def verify_answer(answer, chunks):
    prompt = build_verification_prompt(answer, chunks)
    result = generate_answer(prompt).strip()

    verdict = "NOT SUPPORTED"
    reason = result

    for line in result.splitlines():
        if line.upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    is_supported = "SUPPORTED" in verdict and "NOT" not in verdict
    return is_supported, reason


DISTANCE_THRESHOLD = 1.0  # tune this based on testing - lower = stricter


def ask(question):
    docs, distances = retrieve(question)

    print(f"\nQuestion: {question}")
    print("\n[Retrieved chunks and distances]")
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        preview = doc.strip().replace("\n", " ")[:80]
        print(f"  Chunk {i+1} (distance {dist:.4f}): {preview}...")

    best_distance = min(distances)
    if best_distance > DISTANCE_THRESHOLD:
        print("\n[Answer]")
        print("I don't have enough information in the notes to answer this confidently.")
        print(f"(Best match distance was {best_distance:.4f}, above threshold {DISTANCE_THRESHOLD})")
        print()
        return

    prompt = build_prompt(question, docs)
    answer = generate_answer(prompt)

    print("\n[Answer]")
    print(answer.strip())

    print("\n[Verifying answer against retrieved chunks...]")
    is_supported, reason = verify_answer(answer, docs)

    if is_supported:
        print(f"[VERIFIED] {reason}")
    else:
        print(f"[LOW CONFIDENCE] {reason}")
        print("   (The generated answer may not be fully grounded in your notes.)")
    print()
    


if __name__ == "__main__":
    while True:
        question = input("Ask a question about your notes (or 'quit' to exit): ")
        if question.lower() in ("quit", "exit"):
            break
        ask(question)