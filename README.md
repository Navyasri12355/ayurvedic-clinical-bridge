# Ayurvedic Clinical Bridge - Advanced ML System

A comprehensive Ayurvedic clinical decision support system featuring dual ML architectures (BioBERT + BiLSTM-CRF), explainable AI, and intelligent query processing for bridging traditional Ayurvedic medicine with modern clinical practice.

## 🚀 Quick Start

### Backend (API Server)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the main API server
python main.py
```

<<<<<<< Updated upstream
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
=======
The API will be available at `http://localhost:8000`
>>>>>>> Stashed changes

### Frontend (React Application)
```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Training Models
```bash
# Train BiLSTM-CRF model for fast entity recognition
python scripts/train_bilstm_crf.py --epochs 15 --batch_size 16

# For quick testing with minimal data
python scripts/train_bilstm_crf.py --epochs 2 --batch_size 4 --hidden_dim 64
```

## 📁 Project Structure

```
├── main.py                             # Main FastAPI server
├── ayurvedic_clinical_bridge/          # Core ML services and models
│   ├── models/                         # ML model implementations
│   │   ├── biobert_transformer.py     # BioBERT clinical processor
│   │   ├── bilstm_crf_model.py        # BiLSTM-CRF architecture
│   │   ├── bilstm_crf_processor.py    # BiLSTM-CRF clinical processor
│   │   └── model_manager.py           # Unified model management
│   └── services/                       # Core services
│       ├── herb_predictor.py          # ML-powered herb predictions
│       ├── explainability_service.py  # SHAP-based explanations
│       ├── explanation_manager.py     # Explanation state management
│       ├── query_intent_classifier.py # Query type classification
│       └── enhanced_query_preprocessor.py # Advanced query processing
├── models/                             # Trained model files and configurations
│   ├── biobert/                       # BioBERT model files
│   ├── bilstm_crf/                    # BiLSTM-CRF model files
│   ├── herb_benefits/                 # Herb prediction models
│   └── query_intent_classifier/       # Intent classification models
├── data/                              # Training and reference datasets
│   ├── datasets/                      # Processed training datasets
│   ├── cache/                         # Hugging Face dataset cache
│   ├── amidha_herbs_comprehensive.json # Comprehensive herb database
│   ├── ayurgenix_dataset.csv         # Clinical disease-symptom mappings
│   ├── ayurvedic_herbs_comprehensive.csv # Herb properties and benefits
│   ├── herb.json                     # Core herb reference data
│   └── *.csv, *.json                 # Additional data files
├── scripts/                           # Training and utility scripts
│   └── train_bilstm_crf.py           # BiLSTM-CRF training script
├── frontend/                          # React TypeScript frontend
│   ├── src/
│   │   ├── components/               # React components
│   │   ├── pages/                    # Application pages
│   │   ├── services/                 # API service layer
│   │   └── contexts/                 # React contexts
│   └── package.json
├── config/                            # Configuration files
└── requirements.txt                   # Python dependencies
```

## 🧠 Core Features

### Dual ML Architecture
- **BioBERT Processor**: High-accuracy clinical analysis for practitioners
- **BiLSTM-CRF Processor**: Fast entity recognition for general users
- **Intelligent Model Selection**: Automatic model routing based on user type and query complexity

### Clinical Intelligence
- **Entity Recognition**: Extract herbs, diseases, symptoms, and dosages from clinical text
- **Interaction Detection**: Identify potential herb-drug interactions
- **Safety Assessment**: Comprehensive risk evaluation with confidence scores
- **Treatment Recommendations**: Evidence-based Ayurvedic treatment suggestions

### Explainable AI
- **SHAP Integration**: Token-level and feature-level explanations
- **Multi-Model Explanations**: Explanations for both BioBERT and BiLSTM-CRF
- **Interactive Visualizations**: Frontend components for explanation display
- **Batch Processing**: Efficient explanation generation for multiple queries

### Advanced Query Processing
- **Intent Classification**: Automatic detection of query types (herb benefits, disease prediction, etc.)
- **Query Preprocessing**: Normalization, spell correction, and enhancement
- **Context-Aware Routing**: Intelligent routing to appropriate ML models
- **Multi-Modal Support**: Text, structured data, and clinical note processing

## 🔧 API Endpoints

### Authentication & User Management
- `POST /auth/register` - Register new user
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user profile
- `POST /auth/admin/verify` - Admin verification

### Core Clinical Intelligence
- `GET /health` - API health check and model status
- `POST /predict` - Disease prediction from symptoms
- `POST /api/intelligent-query/process` - Advanced query processing with dual ML
- `POST /api/explainability/explain` - Generate SHAP explanations
- `POST /api/explainability/batch-explain` - Batch explanation processing

### Medicine Mapping & Recommendations
- `POST /api/medicine-mapping/find-alternative` - Find Ayurvedic alternatives
- `POST /api/medicine-mapping/search-by-symptoms` - Symptom-based medicine search
- `POST /api/medicine-mapping/compare-treatments` - Treatment comparison analysis

### Model Management
- `GET /api/models/status` - Get all model statuses
- `POST /api/models/switch` - Switch between BioBERT and BiLSTM-CRF
- `GET /api/models/performance` - Model performance metrics

## 🤖 ML Models & Architecture

### 1. BioBERT Clinical Processor
- **Purpose**: High-accuracy clinical analysis for healthcare practitioners
- **Architecture**: Fine-tuned BioBERT with clinical domain adaptation
- **Capabilities**: 
  - Named Entity Recognition (NER)
  - Relation Extraction
  - Clinical Text Classification
  - Interaction Detection
- **Use Cases**: Detailed clinical analysis, research, practitioner tools

### 2. BiLSTM-CRF Processor
- **Purpose**: Fast entity recognition for general users
- **Architecture**: Bidirectional LSTM with Conditional Random Fields
- **Capabilities**:
  - Real-time entity extraction
  - Sequence labeling
  - Clinical term recognition
- **Use Cases**: Consumer applications, real-time processing, mobile apps

### 3. Specialized Models
- **Herb Predictor** (`models/herb_benefits/`) - ML-powered herb benefit predictions
- **Query Intent Classifier** (`models/query_intent_classifier/`) - Query type detection
- **Disease Predictor** (`models/pure_biobert/`) - Disease classification from symptoms

## 🔍 Explainable AI Features

### SHAP Integration
- **Token-Level Explanations**: Individual word/token contributions
- **Feature Importance**: Global and local feature importance analysis
- **Model Comparison**: Side-by-side explanations for different models
- **Interactive Visualizations**: Web-based explanation viewers

### Explanation Types
- **Prediction Explanations**: Why a specific prediction was made
- **Entity Explanations**: Why entities were recognized
- **Confidence Analysis**: Uncertainty quantification
- **Counterfactual Analysis**: What-if scenario explanations

## 📊 Data & Training

### Key Data Files

#### Core Databases
- **`data/amidha_herbs_comprehensive.json`**: 
  - Comprehensive Ayurvedic herb database
  - Detailed herb properties, benefits, and contraindications
  - Usage guidelines and traditional applications
  - Dosage recommendations and preparation methods

- **`data/ayurgenix_dataset.csv`**: 
  - Clinical disease-symptom mapping dataset
  - 447 diseases with 35 clinical parameters
  - Structured data for disease prediction models
  - Symptom severity and correlation data

- **`data/ayurvedic_herbs_comprehensive.csv`**: 
  - Herb properties in structured CSV format
  - Medicinal properties and therapeutic uses
  - Chemical constituents and active compounds
  - Cross-references with traditional texts

- **`data/herb.json`**: 
  - Core herb reference data
  - Quick lookup for common herbs
  - Essential properties and basic information

### Training Datasets
- **Ayurvedic QA Dataset**: 15M+ tokens of Ayurvedic Q&A pairs
- **Herb Comprehensive Database**: 1000+ herbs with properties and benefits
- **AyurGenix Dataset** (`data/ayurgenix_dataset.csv`): Clinical disease-symptom mappings with 447 diseases and 35 parameters
- **Amidha Herbs Database** (`data/amidha_herbs_comprehensive.json`): Comprehensive herb database with detailed properties, benefits, and usage guidelines
- **Clinical Entity Dataset**: Synthetic NER training data with BIO tagging
- **Query Intent Dataset**: Classified queries for intent detection

### Training Scripts
- `scripts/train_bilstm_crf.py` - Complete BiLSTM-CRF training pipeline
- Synthetic data generation for clinical NER
- Configurable hyperparameters and model architectures
- Evaluation metrics and model persistence

## 🎯 User Interfaces

### General Users Interface
- Simplified query interface
- Fast BiLSTM-CRF processing
- Basic safety recommendations
- Consumer-friendly explanations

### Clinicians Interface
- Advanced BioBERT analysis
- Detailed interaction detection
- Comprehensive safety assessments
- Professional-grade explanations

### Admin Interface
- Model management and switching
- Performance monitoring
- User verification and management
- System configuration

## 🚀 Advanced Features

### Model Comparison
- Side-by-side analysis of BioBERT, BiLSTM-CRF and other simpler models
- Performance benchmarking
- Speed vs accuracy trade-offs
- Use case recommendations

### Intelligent Query Processing
- **Multi-Step Processing**: Query → Intent → Model → Explanation
- **Context Preservation**: Maintains conversation context
- **Adaptive Responses**: Tailored to user expertise level
- **Error Handling**: Graceful degradation and fallback mechanisms

### Safety & Compliance
- **Interaction Warnings**: Herb-drug interaction detection
- **Dosage Validation**: Safe dosage recommendations
- **Contraindication Alerts**: Medical condition warnings
- **Evidence Levels**: Confidence and evidence quality indicators

## 🔧 Configuration & Deployment

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Model Training
```bash
# Train BiLSTM-CRF with custom parameters
python scripts/train_bilstm_crf.py \
  --epochs 15 \
  --batch_size 16 \
  --hidden_dim 256 \
  --embedding_dim 300 \
  --learning_rate 0.001

# Quick training for testing
python scripts/train_bilstm_crf.py --epochs 2 --batch_size 4
```

### Production Deployment
- FastAPI backend with automatic API documentation
- React frontend with TypeScript
- Model versioning and A/B testing support
- Monitoring and logging integration

## 📈 Performance Metrics

### BioBERT Processor
- **Accuracy**: High precision for clinical entity recognition
- **Processing Time**: ~2-5 seconds per query
- **Use Case**: Detailed clinical analysis

### BiLSTM-CRF Processor
- **Speed**: ~50-100ms per query
- **Accuracy**: Good performance for common entities
- **Use Case**: Real-time consumer applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

<<<<<<< Updated upstream

## Disclaimer

This system is for educational and research purposes. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare practitioners for medical decisions.
=======
## 🙏 Acknowledgments

- BioBERT team for the pre-trained clinical language model
- Hugging Face for the transformers library
- SHAP team for explainable AI capabilities
- Ayurvedic medicine practitioners for domain expertise
>>>>>>> Stashed changes
