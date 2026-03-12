# Application Architecture

High-level architecture (simple view).

```mermaid
flowchart TD
  UQ[User Query] --> ORCH[LangGraph Orchestrator]

  subgraph WF[Orchestrated Workflow]
    QR[Query Refinement Agent]
    CG{User Confirmation Gate}
    RA[Retrieval Agent]
    AS[Answer Synthesizer Agent]
    RR[Retry Router]
    RT[Retry Target Node]
    RF[Retry Fallback]
    END([End])
  end

  ORCH --> QR
  QR -->|ok| CG
  QR -->|error| RR

  CG -->|yes| RA
  CG -->|no| END

  RA -->|ok| AS
  RA -->|error| RR

  AS -->|ok| RAGAS[RAGAS Evaluation]
  AS -->|error| RR
  RAGAS --> END

  RR -->|retry_count <= max_retries| RT
  RT --> QR
  RR -->|retry_count > max_retries| RF
  RF --> END
```
