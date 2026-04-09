# LLM Batch Pipeline — Architecture

Visual workflow diagrams for the pipeline stages and data flow.

## Pipeline Flow

```mermaid
flowchart TD
    A[Input Files] --> B[1. Discover]
    B --> C[2. Filter - pre]
    C --> D[3. Transform]
    D --> E[4. Filter - post]
    E --> F[5. Render JSONL]
    F --> G{6. Human Review}
    G -->|Approved| H[7. Submit to Backend]
    G -->|--auto-approve| H
    G -->|Rejected| Z[Abort]
    H --> I[8. Validate Results]
    I --> J[9. Evaluate]
    J --> K[10. Export]

    subgraph Backends
        H --> H1[OpenAI Batch API]
        H --> H2[Ollama Local]
    end

    H1 --> I
    H2 --> I

    subgraph Outputs
        K --> K1[results.xlsx]
        K --> K2[evaluation.xlsx]
        K --> K3[evaluation.json]
        K --> K4[metrics.json]
    end
```

## Plugin System

```mermaid
classDiagram
    class ParsedFile {
        +str filename
        +Path raw_path
        +Any content
        +dict metadata
    }

    class FileReader {
        <<abstract>>
        +can_read(path) bool
        +read(path) ParsedFile
        +package_for_llm(parsed) str
    }

    class Filter {
        <<abstract>>
        +name str
        +apply(parsed) tuple~bool, str~
    }

    class Transformer {
        <<abstract>>
        +name str
        +apply(parsed) ParsedFile
    }

    class OutputTransformer {
        <<abstract>>
        +name str
        +apply(rows) list~dict~
    }

    class PluginSpec {
        +str name
        +FileReader reader
        +list~Filter~ pre_filters
        +list~Transformer~ transformers
        +list~Filter~ post_filters
        +OutputTransformer output_transformer
    }

    PluginSpec --> FileReader
    PluginSpec --> Filter
    PluginSpec --> Transformer
    PluginSpec --> OutputTransformer
    FileReader ..> ParsedFile : creates
    Filter ..> ParsedFile : inspects
    Transformer ..> ParsedFile : transforms
```

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        F1[file_001.eml]
        F2[file_002.eml]
        F3[file_003.eml]
    end

    subgraph "Discover + Filter + Transform"
        P1[ParsedFile]
        P2[ParsedFile]
    end

    subgraph "Render"
        S1[batch-00001.jsonl]
    end

    subgraph "Submit"
        direction TB
        BE{Backend}
        OA[OpenAI]
        OL[Ollama]
    end

    subgraph "Validate + Evaluate"
        V[validated.json]
        E[evaluation.json]
    end

    subgraph "Export"
        X1[results.xlsx]
        X2[evaluation.xlsx]
    end

    F1 --> P1
    F2 --> P2
    F3 -.->|filtered out| DROP[Dropped]
    P1 --> S1
    P2 --> S1
    S1 --> BE
    BE --> OA
    BE --> OL
    OA --> V
    OL --> V
    V --> E
    V --> X1
    E --> X2
```

## Ollama Multi-Server Sharding

```mermaid
flowchart TD
    JSONL[batch.jsonl<br/>1000 requests] --> SHARD[Shard Splitter]

    SHARD --> S1[Shard 1<br/>334 requests]
    SHARD --> S2[Shard 2<br/>333 requests]
    SHARD --> S3[Shard 3<br/>333 requests]

    S1 --> GPU1[GPU Server 1<br/>:11434]
    S2 --> GPU2[GPU Server 2<br/>:11434]
    S3 --> GPU3[GPU Server 3<br/>:11434]

    subgraph "Thread Pool (per shard)"
        GPU1 --> T1A[Thread 1]
        GPU1 --> T1B[Thread 2]
        GPU1 --> T1C[Thread 3]
    end

    T1A --> AGG[Result Aggregator]
    T1B --> AGG
    T1C --> AGG
    GPU2 --> AGG
    GPU3 --> AGG

    AGG --> OUT[output.jsonl]
```

## Batch Directory Structure

```mermaid
graph TD
    ROOT[batches/] --> B1[batch_001_spam_test/]
    ROOT --> B2[batch_002_gdpr_audit/]

    B1 --> CONF[config.toml]
    B1 --> IN[input/]
    B1 --> EVAL_DIR[evaluation/]
    B1 --> JOB[job/]
    B1 --> OUT[output/]
    B1 --> RES[results/]
    B1 --> EXP[export/]
    B1 --> LOGS[logs/]

    IN --> E1[email_001.eml]
    IN --> E2[email_002.eml]

    JOB --> J1[batch-00001.jsonl]
    JOB --> J2[batch.jsonl → batch-00001.jsonl]

    OUT --> O1[output.jsonl]
    OUT --> O2[summary.json]

    RES --> R1[validated.json]

    EXP --> X1[results.xlsx]
    EXP --> X2[evaluation.xlsx]
    EXP --> X3[evaluation.json]

    LOGS --> L1[pipeline.jsonl]
    LOGS --> L2[metrics.json]
```

## Evaluation Output

```mermaid
flowchart LR
    GT[Ground Truth<br/>CSV or prefix map] --> EVAL[Evaluate]
    VR[Validated Results<br/>validated.json] --> EVAL

    EVAL --> CM[Confusion Matrix]
    EVAL --> PM[Per-class Metrics<br/>Precision, Recall, F1]
    EVAL --> ACC[Accuracy]
    EVAL --> ROC[ROC Curve Data<br/>FPR, TPR points]
    EVAL --> AUC[AUC Score]

    CM --> XLSX[evaluation.xlsx]
    PM --> XLSX
    ACC --> XLSX
    ROC --> XLSX
    AUC --> JSON[evaluation.json]
```
