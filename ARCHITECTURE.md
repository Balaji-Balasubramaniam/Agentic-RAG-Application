# Application Architecture

High-level architecture (simple view).

```mermaid
flowchart TD
  U[User + Web UI] --> API[FastAPI API]

  API --> IDX[Indexing\nPDF -> Preprocess -> Embed]
  IDX --> STORE[(Chroma Vector DB + Parent Docstore)]

  API --> ORCH[LangGraph Orchestrator]
  ORCH --> A1[Query Refinement Agent]
  A1 --> A2[Retrieval Agent]
  A2 --> STORE
  STORE --> A3[Answer Synthesizer Agent]
  A3 -.on failure: retry.-> ORCH
  A3 --> RAGAS[RAGAS Evaluation]
  RAGAS --> OK[Final Answer + Sources + Metrics]
  OK --> API
  API --> U
```
