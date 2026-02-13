# Ayurvedic Clinical Bridge - Advanced ML System

A comprehensive Ayurvedic clinical decision support system featuring dual ML architectures (BioBERT + BiLSTM-CRF), explainable AI, and intelligent query processing for bridging traditional Ayurvedic medicine with modern clinical practice.

## 🚀 Quick Start

### Backend (API Server)
The backend is built with **Flask** and provides a RESTful API.

```bash
# Install dependencies
pip install -r requirements.txt
# Note: Ensure flask and flask-cors are installed if not in requirements.txt
pip install flask flask-cors

# Run the main API server
python main.py
# The server will start on port 5000 (default) or 8000 depending on configuration
```

### Frontend (React Application)
The frontend is built with React, TypeScript, and Vite.

```bash
cd frontend
npm install
npm run dev
# The frontend will be available at http://localhost:5173
```

## 📁 Project Structure

```
├── main.py                             # Main Flask server entry point & API routes
├── ayurvedic_clinical_bridge/          # Core ML services and models package
│   ├── models/                         # ML model architectures (BiLSTM-CRF, BioBERT)
│   ├── services/                       # Business logic services
│   │   ├── herb_predictor.py           # Herb benefit prediction
│   │   ├── explainability_service.py   # SHAP explanation service
│   │   └── semantic_herb_matcher.py    # Semantic herb matching
│   └── ...
├── models/                             # Trained model weights and artifacts
│   ├── biobert/                        # BioBERT model files
│   ├── bilstm_crf/                     # BiLSTM-CRF model files
│   └── ...
├── data/                               # Datasets and knowledge bases
│   ├── amidha_herbs_comprehensive.json # Herb database
│   ├── ayurgenix_dataset.csv           # Clinical symptom-disease data
│   └── ...
├── frontend/                           # React TypeScript frontend
│   ├── src/                            # Source code (Components, Pages, Contexts)
│   ├── package.json                    # Frontend dependencies
│   └── vite.config.ts                  # Vite configuration
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## 🧠 Core Features

### Dual ML Architecture
- **BioBERT Processor**: High-accuracy clinical analysis for detailed practitioner insights.
- **BiLSTM-CRF Processor**: Fast named entity recognition (NER) for extracting symptoms, herbs, and diseases.
- **Hybrid Routing**: Intelligent fallback mechanisms to ensure robust response generation.

### Clinical Intelligence
- **Entity Recognition**: Extracts clinical entities (Symptoms, Herbs, Diseases) from natural language text.
- **Disease Prediction**: ML-based prediction of potential diseases based on reported symptoms.
- **Herb Analysis**: Analyzes herb properties (Rasa, Virya, Dosha) and predicts therapeutic benefits.

### Explainable AI (XAI)
- **SHAP Integration**: Provides transparency by explaining *why* a model made a specific prediction.
- **Token-Level Insights**: Highlights which words in the user's query contributed most to the output.

### Medicine Mapping
- **Ayurvedic Alternatives**: capability to map allopathic concepts to Ayurvedic treatments.
- **Symptom Search**: Intelligent search for treatments based on symptoms.

## 🔧 API Endpoints

### Authentication
- `POST /auth/register` - User registration (Supports 'general_user' and 'qualified_practitioner')
- `POST /auth/login` - User login (Returns JWT)
- `GET /auth/me` - Get current user profile

### Intelligent Query & ML
- `POST /api/intelligent-query/process` - Main endpoint for processing natural language queries. Integrates NER, disease prediction, and herb analysis.
- `POST /predict` - Pure ML disease prediction from text.
- `GET /diseases` - List all diseases supported by the model.

### Medicine Mapping
- `POST /api/medicine-mapping/find-alternative` - Find alternatives for medicines.
- `POST /api/medicine-mapping/disease-recommendations` - Get recommendations for specific diseases.
- `POST /api/medicine-mapping/search-by-symptoms` - Search treatments by symptoms.

### Explainability
- `POST /api/explainability/explain` - Generate SHAP explanations for predictions.
- `GET /api/explainability/capabilities` - Check available explainability features.

### System & Health
- `GET /health` - Check API and model loading status.

## ⚠️ Important Notes

- **Backend Framework**: This project uses **Flask**.
- **Data Storage**: User data is currently stored in-memory (`users_db` in `main.py`). This is for demonstration purposes and should be replaced with a persistent database (e.g., PostgreSQL, MongoDB) for production use.
- **Model Dependencies**: The application requires pre-trained models in the `models/` directory to function correctly.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **BioBERT team** for the pre-trained clinical language model.
- **SHAP team** for the explainable AI framework.
