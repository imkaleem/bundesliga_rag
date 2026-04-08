# Bundesliga RAG: Knowledge-Driven Context Retrieval

A Python project that builds **LLM-ready context** for Bundesliga coaching questions by combining **Wikidata (structured knowledge)** and **Wikipedia (unstructured summaries)**.

This repository focuses on the **retrieval layer** of a RAG system: entity resolution, entity linking, provenance, and prompt construction. It intentionally does **not** call an LLM.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Example Output](#example-output)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Project Overview

Given a question like:

- `Who is coaching Berlin?`
- `What about Munich?`
- `Who is Heidenheims manager?`
- `Who is it for Pauli?`

the pipeline:

1. Resolves city/alias mentions to a current Bundesliga club.
2. Retrieves the club's current head coach from Wikidata (`P286`).
3. Retrieves coach background text from Wikipedia (lead summary).
4. Returns a structured prompt string (`SYSTEM`, `USER`, `CONTEXT`) ready for downstream LLM use.

## Key Features

- **Knowledge-driven retrieval** using Wikidata SPARQL + Wikidata API + Wikipedia API
- **Domain-aware entity resolution** for city/club aliases (including multilingual labels)
- **Knowledge-graph traversal** (`club -> head coach`) with explicit Wikidata IDs
- **Provenance-first context** with source references for traceability
- **Robust HTTP behavior** with retries/backoff and clear fallback logic
- **Clean component design** for easy extension into full RAG pipelines

## Architecture

Core data flow:

`User question -> Entity resolution -> Club QID -> Coach QID -> Wikipedia intro -> Prompt assembly`

Main components:

- `BundesligaIndex`: builds and queries city/club alias mappings
- `WikidataClient`: executes SPARQL and entity lookups
- `WikipediaClient`: fetches page summaries
- `RetrievalPipeline`: orchestrates retrieval and builds final prompt

## Repository Structure

- `rag_context_retrieval.py` - main script (console app + retrieval pipeline)
- `requirements.txt` - Python dependencies
- `rag_retrieval.log` - optional runtime log output
- `LICENSE` - license text

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run in interactive mode:

```bash
python rag_context_retrieval.py
```

Run with debug logging:

```bash
python rag_context_retrieval.py --debug --log-file rag_retrieval.log
```

Then type questions in the console:

```text
> Who is coaching Berlin?
```

## Example Output

The script prints a single LLM-ready prompt with:

- `### SYSTEM` - response constraints and citation behavior
- `### USER` - original user question
- `### CONTEXT` - resolved entities, Wikidata IDs, and coach intro text

This output can be passed directly into an LLM call in a separate application layer.

## Design Decisions

- **Entity resolution over generic NER**: domain-specific keys reduce ambiguity.
- **Live retrieval per question**: prioritizes freshness for changing coach assignments.
- **Multilingual normalization**: supports variants such as `Munich` / `Muenchen` / `München`.
- **Fallback strategy**: Wikipedia (`enwiki` -> `dewiki`) then Wikidata description.
- **Provenance and observability**: request-level logs support answer traceability.

## Limitations

- Retrieval scope is intentionally narrow: Bundesliga head coach questions.
- Depends on upstream availability and quality of Wikidata/Wikipedia.
- No caching layer yet (higher latency for repeated similar questions).
- No embedded LLM inference in this repository.

## Roadmap

- Add optional caching for repeated queries.
- Expand to additional relations (e.g., stadium, captain, founded year).
- Add tests for entity resolution and fallback behavior.
- Provide optional API wrapper (FastAPI) for integration into larger systems.

## License

Free to use, modify, and distribute (see `LICENSE`).