# Application Architecture

High-level architecture (simple view).

```mermaid
flowchart TD
  U[User + Web UI] --> API[FastAPI API]

  API --> IDX[Indexing Pipeline\nUpload PDF -> Preprocess -> Embed]
  IDX --> STORE[(Chroma Vector DB + Parent Docstore)]

  API --> ASK[Q&A Pipeline\nRefine -> Confirm if needed -> Retrieve]
  ASK --> STORE
  STORE --> GEN[Generate Answer\nAnswer Synthesizer + Citations]
  GEN --> RAGAS[RAGAS Evaluation]
  RAGAS --> OK[Final Answer + Sources + Metrics]
  OK --> API
  API --> U
```
