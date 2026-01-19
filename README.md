# Ayurvedic Clinical Bridge

An AI-driven clinical decision-support system that bridges modern allopathic medicine with traditional Ayurvedic treatments using advanced NLP models including BiLSTM-CRF and BioBERT. The system provides comprehensive medicine mapping, safety analysis, and clinical knowledge integration for healthcare practitioners and general users.

## Overview

The Ayurvedic Clinical Bridge system bridges modern biomedical practice and traditional Ayurvedic medicine through:

- **Advanced NLP Models**: BiLSTM-CRF for named entity recognition and BioBERT for biomedical text understanding
- **Medicine Mapping**: Comprehensive mapping between allopathic medicines and Ayurvedic alternatives
- **Disease-based Recommendations**: Detailed Ayurvedic treatment protocols for various diseases
- **Safety Analysis**: Herb-drug interaction detection and contraindication checking using biomedical NLP
- **Clinical Knowledge System**: Integrated knowledge base covering 15+ diseases with Ayurvedic treatments
- **Role-based Access**: Different interfaces for general users and qualified practitioners
- **Prescription Analysis**: Entity extraction and treatment recommendation from clinical prescriptions using BioBERT

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prescription  │───▶│  BioBERT +       │───▶│  Medicine       │
│     Input       │    │  BiLSTM-CRF      │    │    Mapper       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Recommendations │◀───│ Safety Analyzer  │◀───│ Knowledge Base  │
│    & Warnings   │    │  (BioBERT)       │    │ (15+ Diseases)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Model Architecture Details

- **BioBERT**: Pre-trained biomedical BERT model for understanding medical terminology and context
- **BiLSTM-CRF**: Bidirectional LSTM with Conditional Random Fields for named entity recognition
- **Hybrid NER**: Combines BioBERT embeddings with BiLSTM-CRF for accurate medical entity extraction
- **Cross-domain Mapping**: Semantic mapping between biomedical and Ayurvedic knowledge domains

## Features

### Core NLP Capabilities
- **BioBERT Integration**: Leverages pre-trained biomedical BERT for understanding medical terminology
- **BiLSTM-CRF NER**: Advanced named entity recognition for extracting medical entities from prescriptions
- **Hybrid Model Architecture**: Combines transformer and RNN approaches for optimal performance
- **Medical Text Processing**: Specialized handling of clinical notes, prescriptions, and medical documents

### Clinical Decision Support
- **Medicine Mapping**: Find Ayurvedic alternatives for allopathic medicines with detailed dosage and preparation instructions
- **Disease Recommendations**: Comprehensive Ayurvedic treatment protocols for 15+ diseases including:
  - Diabetes, Hypertension, Arthritis, Asthma, Migraine
  - Gastritis, Insomnia, Anxiety, Depression, Obesity
  - Anemia, Bronchitis, Constipation, Diarrhea, Eczema
- **Symptom-based Search**: Find potential diseases and treatments based on reported symptoms
- **Safety Analysis**: Herb-drug interaction detection and contraindication checking using biomedical NLP
- **Prescription Analysis**: Extract entities from clinical prescriptions and provide Ayurvedic recommendations
- **Clinical Knowledge Query**: Access integrated knowledge base for disease-specific information

### Model Performance & Comparison
- **BiLSTM vs BioBERT Analysis**: Comparative performance metrics for different model architectures
- **Real-time Processing**: Optimized inference for clinical decision support
- **Confidence Scoring**: Reliability metrics for model predictions
- **Cross-domain Validation**: Accuracy assessment across biomedical and Ayurvedic domains

### Authentication & Authorization
- **JWT-based Authentication**: Secure token-based user authentication
- **Role-based Access Control**: Different access levels for general users and practitioners
- **Practitioner Verification**: Professional credential verification system
- **Admin Panel**: User management and credential approval workflow

### User Interfaces
- **General Users**: Access to medicine mapping and educational information
- **Qualified Practitioners**: Enhanced clinical tools including prescription analysis and safety assessments
- **Medicine Mapping**: Three-tab interface for medicine alternatives, disease recommendations, and symptom search
- **Model Comparison**: Performance analysis comparing BiLSTM-CRF and BioBERT architectures
- **Admin Dashboard**: User management and practitioner verification

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ayurvedic-clinical-bridge
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

#### Option 1: Manual Setup
1. **Start the backend API**
   ```bash
   python run_auth_api.py
   ```

2. **Start the frontend (in another terminal)**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

#### Option 2: Docker Compose
```bash
docker-compose up -d
```

### Authentication System

The system includes comprehensive user authentication with role-based access control:

#### User Types
- **General Users**: Immediate access to medicine mapping and basic information
- **Healthcare Practitioners**: Enhanced access to clinical tools (requires credential verification)
- **Administrators**: User management and system administration

#### Default Access
- Medicine mapping is available without authentication
- Clinical tools require practitioner verification
- Admin functions require administrator privileges

## Project Structure

```
ayurvedic_clinical_bridge/
├── ayurvedic_clinical_bridge/     # Main application package
│   ├── api/                       # REST API endpoints
│   │   ├── main_with_auth.py      # Main API application with authentication
│   │   ├── auth_routes.py         # Authentication endpoints
│   │   ├── medicine_mapping_routes.py  # Medicine mapping API
│   │   ├── prescription_routes.py # Prescription analysis API (BioBERT)
│   │   ├── safety_analysis_routes.py   # Safety analysis API
│   │   ├── knowledge_routes.py    # Clinical knowledge API
│   │   ├── model_routes.py        # Model comparison API (BiLSTM vs BioBERT)
│   │   └── admin_routes.py        # Admin management API
│   ├── models/                    # NLP Models and architectures
│   │   ├── hybrid_ner.py          # BiLSTM-CRF + BioBERT hybrid model
│   │   └── user_models.py         # User authentication models
│   ├── services/                  # Core business logic services
│   │   ├── medicine_mapper.py     # Medicine mapping service
│   │   ├── auth_service.py        # Authentication service
│   │   ├── integrated_knowledge_system_optimized.py  # Knowledge system
│   │   ├── prescription_service_optimized.py         # Prescription analysis (BioBERT)
│   │   ├── safety_analyzer_optimized.py              # Safety analysis
│   │   ├── cross_domain_mapper.py # Biomedical-Ayurvedic mapping
│   │   └── confidence_scorer.py   # Model confidence assessment
│   ├── training/                  # Model training and evaluation
│   │   ├── training_with_metrics.py    # BiLSTM-CRF training
│   │   ├── nlp_metrics.py             # Model evaluation metrics
│   │   └── metrics_collector.py       # Performance data collection
│   ├── middleware/                # Middleware components
│   │   ├── auth_middleware.py     # Authentication middleware
│   │   ├── security.py            # Security utilities
│   │   └── validation.py          # Input validation
│   ├── data/                      # Data processing
│   └── utils/                     # Utility functions
├── frontend/                      # React TypeScript frontend
│   ├── src/
│   │   ├── pages/                 # Application pages
│   │   │   ├── Auth.tsx           # Authentication page
│   │   │   ├── MedicineMapping.tsx # Medicine mapping interface
│   │   │   ├── Clinicians.tsx     # Clinical tools
│   │   │   ├── GeneralUsers.tsx   # General user interface
│   │   │   ├── ModelComparison.tsx # Model comparison
│   │   │   └── AdminVerification.tsx # Admin panel
│   │   ├── contexts/              # React contexts
│   │   └── styles.css             # Application styles
│   ├── package.json               # Frontend dependencies
│   └── vite.config.ts             # Vite configuration
├── data/                          # Data files
│   ├── ayurgenix_dataset.csv      # Main dataset
│   └── datasets/                  # Processed datasets
├── config/                        # Configuration files
├── run_auth_api.py                # Main application entry point
├── docker-compose.yml             # Docker services
├── Dockerfile                     # Application container
└── requirements.txt               # Python dependencies
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Get current user info

### Medicine Mapping
- `POST /api/medicine-mapping/find-alternative` - Find Ayurvedic alternatives for allopathic medicines
- `POST /api/medicine-mapping/disease-recommendations` - Get disease-based Ayurvedic recommendations
- `POST /api/medicine-mapping/search-by-symptoms` - Search treatments by symptoms
- `POST /api/medicine-mapping/check-interactions` - Check herb-drug interactions

### Clinical Tools (Practitioners Only)
- `POST /api/prescription/analyze` - Analyze prescription and extract entities
- `POST /api/safety-analysis/analyze` - Comprehensive safety analysis
- `POST /api/knowledge/query` - Query clinical knowledge base

### Model Comparison
- `GET /api/models/comparison` - Get BiLSTM-CRF vs BioBERT performance comparison
- `GET /api/models/metrics` - Detailed model performance metrics
- `POST /api/models/evaluate` - Evaluate model performance on custom data

### Admin (Administrators Only)
- `GET /api/admin/pending-verifications` - Get pending practitioner verifications
- `POST /api/admin/verify-practitioner` - Verify practitioner credentials

## Key Features in Detail

### Advanced NLP Processing
The system employs state-of-the-art NLP models for medical text processing:

1. **BioBERT Integration**: Pre-trained on biomedical literature for understanding medical terminology
2. **BiLSTM-CRF Architecture**: Bidirectional LSTM with CRF layer for sequence labeling and NER
3. **Hybrid Model Approach**: Combines transformer and RNN architectures for optimal performance
4. **Cross-domain Knowledge Mapping**: Semantic bridging between biomedical and Ayurvedic domains

### Medicine Mapping
The medicine mapping system provides three main functionalities:

1. **Medicine Alternatives**: Find Ayurvedic alternatives for specific allopathic medicines
2. **Disease Recommendations**: Get comprehensive Ayurvedic treatment protocols for diseases
3. **Symptom Search**: Search for potential diseases and treatments based on symptoms

### Model Performance Analysis
Comprehensive comparison between different NLP architectures:
- **BiLSTM-CRF**: Optimized for named entity recognition in medical texts
- **BioBERT**: Leverages transformer architecture for contextual understanding
- **Hybrid Approach**: Combines both models for enhanced accuracy
- **Performance Metrics**: Precision, recall, F1-score, and inference time comparisons

### Disease Coverage
The system provides detailed Ayurvedic treatment information for:
- **Metabolic**: Diabetes, Obesity
- **Cardiovascular**: Hypertension
- **Musculoskeletal**: Arthritis
- **Respiratory**: Asthma, Bronchitis
- **Neurological**: Migraine, Anxiety, Depression, Insomnia
- **Digestive**: Gastritis, Constipation, Diarrhea
- **Hematological**: Anemia
- **Dermatological**: Eczema

### Safety Analysis
Comprehensive safety analysis powered by biomedical NLP includes:
- Herb-drug interaction detection using BioBERT
- Contraindication checking with medical knowledge graphs
- Dosage recommendations based on clinical guidelines
- Precautions and warnings with confidence scoring
- Risk assessment based on patient profile and medical history

## Development

### Frontend Development
The frontend is built with React + TypeScript + Vite:

```bash
cd frontend
npm run dev     # Development server
npm run build   # Production build
npm run preview # Preview production build
```

### Backend Development
The backend uses FastAPI with Python:

```bash
# Development with auto-reload
python run_auth_api.py

# Or with uvicorn directly
uvicorn ayurvedic_clinical_bridge.api.main_with_auth:app --reload
```

### Environment Variables
Key environment variables (see `.env.example`):
- `SECRET_KEY`: JWT secret key
- `DATABASE_URL`: Database connection string
- `VITE_BACKEND_URL`: Backend URL for frontend (default: http://localhost:8000)

### Code Quality
- **Formatting**: Black (Python), Prettier (TypeScript)
- **Linting**: Flake8 (Python), ESLint (TypeScript)
- **Type Checking**: MyPy (Python), TypeScript compiler

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks:
   ```bash
   # Python
   black . && flake8 && mypy .
   
   # Frontend
   cd frontend && npm run lint && npm run type-check
   ```
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Data Sources

The system integrates knowledge from multiple sources using advanced NLP processing:
- **Ayurgenix Dataset**: Primary dataset with 15+ diseases and Ayurvedic treatments
- **Biomedical Literature**: Processed using BioBERT for medical knowledge extraction
- **Traditional Texts**: Classical Ayurvedic literature with NLP-based knowledge extraction
- **Clinical Guidelines**: Evidence-based treatment protocols processed through BiLSTM-CRF
- **Medical Ontologies**: Structured medical knowledge for cross-domain mapping

## Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Considerations
- Set strong `SECRET_KEY` in production
- Use HTTPS for all communications
- Configure proper CORS settings
- Set up database backups
- Monitor API rate limits
- Implement proper logging and monitoring


## Disclaimer

This system is for educational and research purposes. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare practitioners for medical decisions.
