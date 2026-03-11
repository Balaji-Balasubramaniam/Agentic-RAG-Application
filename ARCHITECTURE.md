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
  GEN --> SAFE{RAGAS Faithfulness Gate}
  SAFE -->|Pass| OK[Final Answer + Sources]
  SAFE -->|Fail| FB[Fallback Response]

  OK --> API
  FB --> API
  API --> U
```
