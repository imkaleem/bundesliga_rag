# Bundesliga RAG – Knowledge-Driven Context Retrieval

A **knowledge-driven context retrieval pipeline** for a Retrieval-Augmented Generation (RAG) chatbot. It combines **structured knowledge** (Wikidata, SPARQL) with **unstructured text** (Wikipedia) to answer colloquial questions about **head coaches of clubs in Germany’s 1. Bundesliga**. The pipeline demonstrates **entity resolution**, **entity linking**, **knowledge-graph traversal** (following relations such as P286 head coach), and **provenance-aware retrieval**—core skills for knowledge engineering and RAG systems.

---

## 1. Overview

Given a colloquial user question like:

- “Who is coaching Berlin?”
- “What about munich?”
- “Who is heidenheims manager?”
- “Who is it for Pauli?”

…the script will:

1. **Resolve** the mentioned **city** (or the alias **Pauli**) to the Bundesliga club in that city.
2. Query **Wikidata** (via **SPARQL**, every question) for the **current head coach**.
3. Query **Wikipedia** (every question) for the **intro** (lead summary) of the coach’s article.
4. Output a **single prompt string** containing:
   - a **system prompt** for an LLM,
   - the **user question**,
   - the **retrieved context** (club + coach + coach intro).

> The actual LLM call is **not** implemented (not required by the challenge).

---

## 2. Key Design Choices (Knowledge Engineering View)

### 2.1 Domain model (implicit KG pattern)
The script treats the world as a small knowledge graph slice:

**City → Bundesliga club → Head coach → Coach description**

- “City → club” is handled via a **Bundesliga index** built from Wikidata.
- “Club → coach” is retrieved live **on every question** from Wikidata (`P286 head coach`).
- “Coach → description” is retrieved live **on every question** from the coach’s Wikipedia lead.

This meets the challenge requirement: **entity identification → entity linking → inference → retrieval**.

### 2.2 Entity linking in practice
Instead of generic **NER**, the script uses a *domain-aware entity linking approach*:

- Build a **dictionary/index** of supported **cities** and **club aliases** from Wikidata.
- Normalize user input (**case-insensitive**, strip diacritics: *München → Munchen*).
- Match the **longest** city/alias key in the question (handles *“heidenheims”*).
- Handle the special assumption: **“Pauli” → FC St. Pauli** (if present in current Bundesliga).

---

## 3. Technologies, Skills & Design

| Area | Implementation |
|------|----------------|
| **Structured KB** | Wikidata (SPARQL over `wd:` / `wdt:`); `wbgetentities` for labels, descriptions, sitelinks |
| **Entity resolution** | Domain index from Wikidata (clubs, cities, alt labels); longest-match key lookup; normalized text (case, diacritics). Labels/altLabels in **en** and **de** so e.g. *Munich* and *München* both resolve. |
| **Entity linking** | City/club mention → club QID; club QID → head coach QID via `P286` |
| **Unstructured retrieval** | Wikipedia REST summary API (lead / intro) for coach descriptions; fallback chain: enwiki → dewiki → Wikidata description |
| **Provenance** | Context block includes resolved club/coach QIDs and Wikipedia article references; system prompt instructs LLM to cite sources. Each run has a **request_id**; debug logs capture matched key, QIDs, SPARQL, HTTP, and fallbacks so you can trace *“Why did the chatbot answer X?”* |
| **Robustness** | HTTP retries (429, 5xx) and configurable **User-Agent** (Wikimedia etiquette). On resolution failure: prompt includes error state and list of supported cities; on upstream failure: structured error and instruction to clarify/retry. Optional `--debug --log-file` for full trace. |

**Design notes:** Clear separation of concerns (`BundesligaIndex`, `WikidataClient`, `WikipediaClient`, `RetrievalPipeline`) keeps the pipeline easy to extend into a full RAG stack.

---


## 4. Setup

**Requirements:** Python 3.8+

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the script**

   ```bash
   python rag_context_retrieval.py
   ```

   Optional debug logging:

   ```bash
   python rag_context_retrieval.py --debug --log-file rag_retrieval.log
   ```

3. **Use:** Type a question at the prompt; the script prints the **LLM-ready prompt** (system + user + context).

   ```
   > Who is coaching Berlin?
   ```

---

## 5. Output Format (Prompt)

The pipeline outputs a **single prompt string** in three sections:

- **`### SYSTEM`** – Instructions to the LLM (scope, use-only-context, cite provenance).
- **`### USER`** – The user’s question as entered.
- **`### CONTEXT`** – Resolved club (with QID), current head coach (with QID), coach Wikipedia intro, and source references.

This string can be passed directly to an LLM in a RAG stack. The **LLM invocation is not included** in this repo (out of scope for the retrieval component).

---

## 6. Additional Questions (Answers)

### 6.1 Advantages and disadvantages of using additional information (RAG) vs letting the LLM answer without it

**Advantages:** **Grounding** the LLM with external data yields more **accurate, up-to-date** answers. For example, RAG can incorporate **current facts** beyond the model's **training cutoff**. This **reduces hallucinations** and improves **domain-specific correctness**.

**Disadvantages:** It adds **system complexity** and **latency**. The pipeline must **compute embeddings**, **perform searches**, and **manage indexes**, which can slow down response time. There are also **extra costs** (API calls, vector DB operations) and **maintenance overhead**.

**LLM-only** (no extra data): **+** Simple to deploy and usually **faster** (no retrieval step). **–** But its knowledge is **static** (fixed at training time), so it may produce **outdated** or **fabricated** answers on recent data. 

---

### 6.2 Advantages and disadvantages of querying this data on every user question

**Advantages:** **Always fetching fresh data** ensures the chatbot answers with the **latest information** (e.g. any coach changes are **immediately reflected**). There's **no risk of staleness** from cached data.

**Disadvantages:** Each question incurs **overhead** (SPARQL and HTTP calls), increasing **latency** and **dependence on external services**. Frequent queries can hit **rate limits** or **timeouts**. If the data **rarely changes** (coaches don't switch often), it may be more efficient to **cache** or **periodically update** the index to reduce repeat queries.

---

### 6.3 How would the process change if the information about coaches only were available via PDF?

Instead of **online APIs**, we would need to **ingest** the PDF content into our retrieval system. This means **extracting text** (e.g. using a PDF library like **PyPDF2** or **OCR**), **chunking** it into passages, and **building embeddings** for those chunks. The chatbot would then **search** this **PDF-derived corpus** for answers. In other words, we'd add a **preprocessing step**: **extract and index** the PDF text into the **vector store**, then proceed with retrieval the same way. This adds **complexity** (parsing, chunking), but allows the RAG pipeline to work with the PDF as its **knowledge base**.

---

### 6.4 Do you see potential for agents in this process? If so, where and how?

**Yes.** An **"agentic RAG"** design could **orchestrate** the workflow. For instance, an **AI agent** could **parse** the question, **decide** which **tools** or **knowledge sources** to call (Wikidata vs Wikipedia), **execute** those calls, and **iteratively refine** the query. Agentic RAG systems use **planning** and **tool-calling**: e.g. **breaking a task into sub-queries**, **retrieving from multiple sources**, and **chaining results**. This can improve **adaptability** and handle **more complex questions**, as agents **manage** the retrieval steps and **maintain context** over time.


---

### 6.5 How do these processes profit from a domain data model?

**Modeling the domain** (clubs, cities, coaches) in a **structured way** pays off in **accuracy** and **consistency**. A **knowledge graph** or **schema** captures exactly **which city belongs to which club** and **who the coach is**. This **prevents ambiguity** (e.g. which “Berlin” is meant) and **ensures correctness**. As one source notes, knowledge graphs **“ensure consistency”** and allow **“complete and correct aggregation of facts”**. In practice, a structured model gives the LLM a **“rich, connected context”** instead of loose text snippets. **Grounding** answers in a **domain graph** **reduces hallucination** and supports **multi-hop reasoning** across related entities. By **encoding domain facts** as **data model properties**, the retrieval stage returns **precise facts** rather than isolated text, improving the chatbot’s **reliability** and **coherence**.

---

## 7. File Overview

- **`rag_context_retrieval.py`** – main script (console interface, retrieval pipeline)
- `requirements.txt` – dependencies
- `rag_retrieval.log` – generated on execution (if enabled)

---

## LICENSE
Free to use, modify, and distribute.