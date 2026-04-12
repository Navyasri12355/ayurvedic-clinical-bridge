# Ayurvedic Clinical Bridge - Advanced ML System

A comprehensive Ayurvedic clinical decision support system featuring dual ML architectures (BioBERT + BiLSTM-CRF), explainable AI, and intelligent query processing for bridging traditional Ayurvedic medicine with modern clinical practice.

## Quick Start

### Backend (API Server)
The backend is built with Flask and provides a RESTful API.

```bash
# Install dependencies
pip install -r requirements.txt
# Note: Ensure flask and flask-cors are installed if not in requirements.txt
pip install flask flask-cors

# Run the main API server
python main.py
# The server will start on http://0.0.0.0:8000
```

### Frontend (React Application)
The frontend is built with React, TypeScript, and Vite.

```bash
cd frontend
npm install
npm run dev
# The frontend will be available at http://localhost:5173
```

## Project Structure

```
├── main.py                             # Main Flask server entry point & API routes
├── pyproject.toml                      # Project configuration
├── requirements.txt                    # Python dependencies
├── ayurvedic_clinical_bridge/          # Core ML services and models package
│   ├── __init__.py
│   ├── api/                            # API route definitions
│   ├── models/                         # ML model architectures (BiLSTM-CRF, BioBERT)
│   │   ├── bilstm_crf_model.py         # BiLSTM-CRF model
│   │   ├── bilstm_crf_processor.py     # BiLSTM-CRF processor
│   │   ├── biobert_transformer.py      # BioBERT model
│   │   ├── hybrid_ner.py               # Hybrid NER
│   │   └── model_manager.py            # Model management
│   └── services/                       # Business logic services
│       ├── enhanced_query_preprocessor.py
│       ├── explainability_service.py   # SHAP explanation service
│       ├── explanation_manager.py
│       ├── herb_predictor.py           # Herb benefit prediction
│       ├── herb_synonym_matcher.py
│       ├── query_intent_classifier.py
│       └── semantic_herb_matcher.py    # Semantic herb matching
├── config/                             # Configuration module
│   └── settings.py
├── data/                               # Datasets and knowledge bases
│   ├── amidha_herbs_comprehensive.json # Herb database
│   ├── herb.json
│   └── datasets/                       # Processed datasets
│       ├── ayurgenixai_processed.csv
│       ├── ayurvedic_meals_processed.csv
│       ├── ayurvedic_qa_processed.csv
│       └── ayurvedic_remedies_processed.csv
├── models/                             # Trained model weights and artifacts
│   ├── bilstm_crf/                     # BiLSTM-CRF model files
│   ├── herb_benefits/                  # Herb benefits model
│   ├── pure_biobert/                   # BioBERT model files
│   └── query_intent_classifier/        # Intent classifier model
├── frontend/                           # React TypeScript frontend
│   ├── src/                            # Source code
│   │   ├── components/                 # Reusable components
│   │   ├── pages/                      # Page components
│   │   ├── contexts/                   # React contexts
│   │   ├── services/                   # Frontend services
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── public/
├── scripts/                            # Training and utility scripts
│   ├── train_bilstm_crf.py
│   └── train_models.py
└── README.md                           # Project documentation
```

## Core Features

### Dual ML Architecture
- BioBERT Processor: High-accuracy clinical disease prediction trained on Ayurvedic datasets.
- BiLSTM-CRF Processor: Fast named entity recognition (NER) for extracting symptoms, herbs, and diseases.
- Hybrid Routing: Intelligent query processing combining both models for robust predictions.

### Clinical Intelligence
- Entity Recognition: Extracts clinical entities (Symptoms, Herbs, Diseases) from natural language text using BiLSTM-CRF.
- Disease Prediction: ML-based prediction of potential diseases based on reported symptoms using BioBERT.
- Herb Analysis: Analyzes herb properties and predicts therapeutic benefits using trained herb benefit models.
- User Role Awareness: Different response formatting for general users versus qualified practitioners.

### Explainable AI (XAI)
- SHAP Integration: Provides transparency by explaining why a model made a specific prediction.
- Token-Level Insights: Highlights which words in the user's query contributed most to the output.
- Global Feature Importance: Analyzes which clinical terms are most important overall in the model.
- Batch Explanations: Process multiple predictions with explanations simultaneously.

## API Endpoints

### Authentication
- POST /auth/register - User registration (Supports 'general_user' and 'qualified_practitioner')
- POST /auth/login - User login (Returns JWT)
- GET /auth/me - Get current user profile

### Intelligent Query & ML
- POST /api/intelligent-query/process - Main endpoint for processing natural language queries. Integrates NER, disease prediction, and herb analysis. Requires authentication.
- GET /api/intelligent-query/capabilities - Get available models and supported query types.
- GET /api/intelligent-query/health - Get intelligent query system status.
- POST /predict - Pure ML disease prediction from text.
- GET /diseases - List all diseases supported by the model.

### Model Information
- GET /api/models/comparison - Get model comparison metrics and benchmarks.
- GET /api/models/training-status - Get training status of models.
- POST /api/models/trigger-training - Trigger model training (placeholder endpoint).

### Explainability
- POST /api/explainability/explain - Generate explanations for predictions using SHAP. Requires authentication.
- GET /api/explainability/global-importance - Get global feature importance analysis. Requires authentication.
- POST /api/explainability/batch-explain - Batch process explanations for multiple texts. Maximum 10 texts per batch. Requires authentication.
- GET /api/explainability/capabilities - Check available explainability features.

### System and Health
- GET /health - Check API and model loading status.

## Important Notes

- Backend Framework: This project uses Flask.
- Data Storage: User data is currently stored in-memory (users_db in main.py). For production use, replace with a persistent database (e.g., PostgreSQL, MongoDB).
- Model Dependencies: The application requires pre-trained models in the models/ directory. If models are missing, the corresponding features will be unavailable but the API will still function.
- Pure ML Approach: The system uses only trained ML models with no keyword lists, rule-based heuristics, or hard-coded mapping tables.
- Authentication: Most endpoints return 401 status without a valid JWT token for authenticated endpoints marked with require_auth decorator.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- BioBERT team for the pre-trained clinical language model.
- SHAP team for the explainable AI framework.
