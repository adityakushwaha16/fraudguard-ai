# Diagrams (Mermaid - render in VS Code with "Markdown Preview Mermaid Support", GitHub, or mermaid.live)

## System architecture
```mermaid
flowchart LR
  U[User browser] -->|HTML/JS/Chart.js| F[Flask templates + static]
  F -->|REST JSON| API[Flask API / routes]
  API --> V[Input validation]
  V --> P[Saved sklearn Pipeline]
  P --> M[Trained model]
  M --> T[Decision threshold]
  M --> S[SHAP explainer]
  T --> DB[(SQLite / PostgreSQL)]
  S --> DB
  DB --> API --> F
  subgraph Offline training
    D[(PaySim CSV)] --> E[ml/eda.py] --> TR[ml/train.py] --> A[(models/*.joblib, JSON)]
  end
  A -. loaded once .-> P
```

## Data flow diagram (Level 1)
```mermaid
flowchart TD
  User -->|credentials| P1[1 Authenticate]
  P1 <--> D1[(Users)]
  User -->|transaction fields| P2[2 Validate + preprocess]
  P2 --> P3[3 Predict + SHAP]
  P3 <--> D3[(Model artifacts)]
  P3 --> P4[4 Store result]
  P4 --> D2[(Transactions + Predictions)]
  D2 --> P5[5 Dashboard / analytics / history]
  P5 --> User
```

## Use case diagram
```mermaid
flowchart LR
  actor((User))
  actor --- r[Register / Login]
  actor --- a[Analyze transaction]
  actor --- v[View risk + SHAP explanation]
  actor --- h[Search / filter history]
  actor --- d[View dashboard + analytics]
  actor --- s[Monitor suspicious transactions]
  actor --- p[View profile]
  mt((ML engineer)) --- t[Train / evaluate / export model]
```

## ER diagram
```mermaid
erDiagram
  USERS ||--o{ TRANSACTIONS : submits
  TRANSACTIONS ||--|| PREDICTIONS : has
  USERS { int id PK string full_name string email UK string password_hash datetime created_at }
  TRANSACTIONS { int id PK int user_id FK int hour_of_day string type float amount float oldbalance_org float oldbalance_dest datetime created_at }
  PREDICTIONS { int id PK int transaction_id FK,UK float fraud_probability string label string risk_level float threshold_used text explanation_json string model_version datetime created_at }
```

## Sequence: analyze a transaction
```mermaid
sequenceDiagram
  participant B as Browser
  participant API as Flask API
  participant PR as Predictor
  participant DB as Database
  B->>API: POST /api/transactions/predict
  API->>API: validate_tx()
  API->>PR: predict(clean)
  PR->>PR: engineer_features + pipeline.predict_proba
  PR->>PR: threshold + risk level + SHAP
  PR-->>API: result
  API->>DB: insert Transaction + Prediction
  API-->>B: 201 JSON
  B->>B: redirect to /transactions/{id}
```
