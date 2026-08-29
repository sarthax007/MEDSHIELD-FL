# MedShield-FL Database Schema

This document outlines the PostgreSQL database schema for the MedShield-FL central server. 
The database is designed to store metadata regarding hospitals, training rounds, models, and predictions, while strictly avoiding the central storage of raw patient data, in alignment with our privacy-preserving architecture.

## Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    Hospital ||--o{ User : "has"
    Hospital ||--o{ ImageMetadata : "owns"
    
    TrainingRound ||--o{ ModelVersion : "produces"
    
    ModelVersion ||--o{ Prediction : "makes"
    
    ImageMetadata ||--o{ Label : "receives"
    ImageMetadata ||--o{ Prediction : "targeted by"
    
    User ||--o{ Label : "submits"
    User ||--o{ AuditLog : "performs"
    
    Hospital {
        uuid id PK
        string name
        string location
        timestamp created_at
    }

    User {
        uuid id PK
        uuid hospital_id FK
        string username
        string hashed_password
        string role
        timestamp created_at
    }

    TrainingRound {
        int id PK
        int round_number
        string status
        timestamp start_time
        timestamp end_time
        float global_accuracy
    }

    ModelVersion {
        uuid id PK
        string version_tag
        int training_round_id FK
        string s3_path
        timestamp created_at
    }

    ImageMetadata {
        uuid id PK
        uuid hospital_id FK
        string local_image_ref
        string modality
        string status
        timestamp created_at
    }

    Label {
        uuid id PK
        uuid image_id FK
        uuid user_id FK
        string class_label
        timestamp submitted_at
    }

    Prediction {
        uuid id PK
        uuid image_id FK
        uuid model_version_id FK
        string predicted_class
        float confidence
        timestamp created_at
    }

    AuditLog {
        uuid id PK
        uuid user_id FK
        string action
        string resource_type
        uuid resource_id
        timestamp timestamp
        json details
    }
```

## Schema Details

### Primary Keys
- All tables use **UUID (UUIDv4)** for primary keys (`id`), EXCEPT for the `TrainingRound` table, which uses an **auto-incrementing integer** (`id`) to allow for sequential tracking of rounds easily.

### Privacy & Data Locality Constraints
- **No Raw Images**: The `ImageMetadata` table deliberately excludes any `bytea` or BLOB data columns. Images are referenced by a string (`local_image_ref`), representing the image's location or identifier within the local hospital's secure environment. The central server has no access to the image pixels.

### Relationships
- **Hospital - User**: One hospital can have multiple users (admins, doctors, operators).
- **Hospital - ImageMetadata**: A hospital registers metadata for its images to be part of active learning or inference pools.
- **TrainingRound - ModelVersion**: A training round produces model versions (often a single global model per round).
- **ImageMetadata - Label**: An image can receive labels (from doctors) which flow into the Active Learning loop.
- **ImageMetadata - Prediction**: Predictions are made on specific images using specific model versions.
- **User - AuditLog**: Important user actions are tracked in the audit log for compliance.
