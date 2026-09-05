# Glass-Box RAG: A Self-Verifying Retrieval-Augmented Generation System

A local RAG application that doesn't just answer questions from your documents — it shows
you exactly what it retrieved, and double-checks its own answer against the source material
before presenting it as fact.

## The Problem

Most RAG demos hide the retrieval step and trust the LLM's output blindly. This means a
system can retrieve weak or irrelevant context and still generate a confident, wrong answer,
with no visible warning to the user. This project treats that as the core problem to solve,
not an edge case to ignore.

## What It Does

1. **Ingests** a PDF (Computer Networks / ML lecture notes), extracts text, and splits it
   into sentence-aware chunks (no mid-word cuts).
2. **Embeds** each chunk and stores it in a local Chroma vector database.
3. **Retrieves** the most relevant chunks for a user's question via similarity search.
4. **Refuses to answer** upfront if even the best-matching chunk is too dissimilar
   (distance above a tuned threshold) — instead of forcing an answer from irrelevant context.
5. **Generates** an answer using a local LLM (Llama 3.2 via Ollama), constrained by prompt
   rules to only use the provided context and cite which chunk(s) it used.
6. **Self-verifies**: sends the generated answer back through the LLM as a strict fact-checker,
   comparing it against the retrieved context, and labels the answer Verified or Low Confidence.
7. **Displays all of this** in a Streamlit UI — question, retrieved chunks (collapsible),
   generated answer, and a confidence badge.

## Architecture

PDF → pdfplumber (text extraction) → sentence-based chunking
→ Chroma (embed + store) → similarity search (top-k retrieval)
→ distance threshold check → Ollama/Llama 3.2 (grounded answer generation)
→ Ollama/Llama 3.2 (self-verification pass) → Streamlit UI


## Tech Stack
- **Python** — pipeline and app logic
- **pdfplumber** — PDF text extraction
- **ChromaDB** — local vector database
- **Ollama (Llama 3.2)** — local LLM for generation and verification, no API cost
- **Streamlit** — UI

## Evaluation

Tested across 4 categories of questions against the source document:

| # | Question | Category | Best Distance | Result | Correct? |
|---|----------|----------|---------------|--------|----------|
| 1 | What is TCP? | In-scope, simple | 0.79 | Verified | Yes |
| 2 | What is the OSI model? (layer order) | In-scope, multi-part | 0.86 | Flagged Low Confidence | Correctly flagged an ordering error |
| 3 | What is a router? (pre-extraction-fix) | In-scope, simple | 1.24 | Flagged Low Confidence | False negative — traced to a text-extraction bug |
| 4 | What is a router? (post-fix) | In-scope, simple | 1.06 | Verified | Yes |
| 5 | What is ensemble learning? | In-scope, simple | 0.79 | Verified | Yes |
| 6 | Compare Hub, Switch, Router | In-scope, multi-part | 0.71 | Verified | No — verifier's own stated reasoning contradicted its verdict |
| 7 | What is a black hole? | Out-of-scope | N/A | Correctly refused | Yes |

## Key Engineering Decisions & Bugs Found

**1. Text extraction bug → false verification negatives**
Initial `pdfplumber` extraction merged words together on this PDF's layout (e.g.
`ConnectsLANtoInternet`). This caused the verifier to wrongly flag a correct answer as
unsupported. Fixed by tuning `x_tolerance` in `extract_text()`.

**2. Chunking strategy: character-based → sentence-based**
Fixed-character-count chunking frequently cut chunks mid-word or mid-sentence. Switched to
sentence-boundary-aware chunking, which fixed fragment issues and improved retrieval distances.

**3. Retrieval always returns *something*, even when nothing is relevant**
Added a distance threshold check before generation — if even the best match exceeds the
threshold, the system refuses to answer rather than forcing an answer from irrelevant text.

## Known Limitations

- **The verifier can contradict its own reasoning.** In testing, it correctly stated in its
  reasoning that "Hub is Layer 1, Switch is Layer 2" while still marking an answer that
  claimed both were Layer 1 as "Verified" — small local models aren't fully reliable self-critics.
- **Distance threshold is manually tuned**, not adaptive.
- **Fixed-size sentence grouping** can still mix unrelated adjacent content.

## What I'd Improve Next
- Use a larger model (e.g. Llama 3.1 8B) specifically for verification.
- Move to semantic/section-based chunking using document headings.
- Build a labeled eval set to track verifier accuracy over time.

## Running It Locally
```bash
pip install -r requirements.txt
ollama pull llama3.2
python ingest.py        # one-time: builds the vector database
streamlit run app.py    # launches the UI
```