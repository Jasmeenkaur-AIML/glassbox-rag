import streamlit as st
import chromadb
import pdfplumber
import re
from groq import Groq

MODEL = "openai/gpt-oss-20b"
DISTANCE_THRESHOLD = 1.15
PDF_PATH = "data/notes.pdf"
CHUNK_SIZE = 500

st.set_page_config(page_title="Glass-Box RAG", page_icon="📘")

client_groq = Groq(api_key=st.secrets["GROQ_API_KEY"])


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1.5, y_tolerance=3)
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        return client.get_collection("notes")
    except Exception:
        with st.spinner("First-time setup: processing notes.pdf..."):
            text = extract_text(PDF_PATH)
            chunks = chunk_text(text)
            collection = client.create_collection("notes")
            collection.add(
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
        return collection


collection = get_collection()


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
    response = client_groq.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


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


st.title("📘 Glass-Box RAG")
st.caption("Ask questions about your notes. See exactly what was retrieved, and whether the answer was verified.")

question = st.text_input("Ask a question about your notes:")

if question:
    with st.spinner("Retrieving relevant chunks..."):
        docs, distances = retrieve(question)
    best_distance = min(distances)

    with st.expander("🔍 Retrieved chunks (click to expand)", expanded=False):
        for i, (doc, dist) in enumerate(zip(docs, distances)):
            st.markdown(f"**Chunk {i+1}** — distance: `{dist:.4f}`")
            st.text(doc.strip()[:400])
            st.divider()

    if best_distance > DISTANCE_THRESHOLD:
        st.warning("I don't have enough information in the notes to answer this confidently.")
        st.caption(f"Best match distance was {best_distance:.4f}, above threshold {DISTANCE_THRESHOLD}")
    else:
        with st.spinner("Generating answer..."):
            prompt = build_prompt(question, docs)
            answer = generate_answer(prompt)

        with st.spinner("Verifying answer against sources..."):
            is_supported, reason = verify_answer(answer, docs)

        st.subheader("Answer")
        st.write(answer.strip())

        if is_supported:
            st.success(f"✅ Verified — {reason}")
        else:
            st.error(f"⚠️ Low confidence — {reason}")