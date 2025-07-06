# Agent Selection Diagram

```mermaid
graph LR
    A[User Input] --> B[Intent Analysis]
    B --> C{Confidence?}
    
    C -->|0.95| D[Climate Agent<br/>set temperature]
    C -->|0.90| E[Music Agent<br/>play music]
    C -->|0.95| F[Vehicle Agent<br/>lock doors]
    C -->|0.85| G[Navigation Agent<br/>find temples]
    C -->|0.20| H[Groq AI<br/>general chat]
    
    D --> I[Vehicle Tools]
    E --> I
    F --> I
    G --> J[External APIs]
    H --> K[AI Response]
    
    I --> L[Update State]
    J --> L
    K --> M[Response to User]
    L --> M
    
    style A fill:#e1f5fe
    style C fill:#fff3e0
    style D fill:#e8f5e8
    style E fill:#e8f5e8
    style F fill:#e8f5e8
    style G fill:#e8f5e8
    style H fill:#ffebee
    style M fill:#e1f5fe
