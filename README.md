# Ayurvedic Clinical Bridge

An AI-driven clinical decision-support system that bridges modern allopathic medicine with traditional Ayurvedic treatments. The system provides comprehensive medicine mapping, safety analysis, and clinical knowledge integration for healthcare practitioners and general users.

## Overview

The Ayurvedic Clinical Bridge system bridges modern biomedical practice and traditional Ayurvedic medicine through:

- **Medicine Mapping**: Comprehensive mapping between allopathic medicines and Ayurvedic alternatives
- **Disease-based Recommendations**: Detailed Ayurvedic treatment protocols for various diseases
- **Safety Analysis**: Herb-drug interaction detection and contraindication checking
- **Clinical Knowledge System**: Integrated knowledge base covering 15+ diseases with Ayurvedic treatments
- **Role-based Access**: Different interfaces for general users and qualified practitioners
- **Prescription Analysis**: Entity extraction and treatment recommendation from clinical prescriptions

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prescription  │───▶│  Entity Parser   │───▶│  Medicine       │
│     Input       │    │  & Analyzer      │    │    Mapper       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Recommendations │◀───│ Safety Analyzer  │◀───│ Knowledge Base  │
│    & Warnings   │    │                  │    │ (15+ Diseases)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

### Core Capabilities
- **Medicine Mapping**: Find Ayurvedic alternatives for allopathic medicines with detailed dosage and preparation instructions
- **Disease Recommendations**: Comprehensive Ayurvedic treatment protocols for 15+ diseases including:
  - Diabetes, Hypertension, Arthritis, Asthma, Migraine
  - Gastritis, Insomnia, Anxiety, Depression, Obesity
  - Anemia, Bronchitis, Constipation, Diarrhea, Eczema
- **Symptom-based Search**: Find potential diseases and treatments based on reported symptoms
- **Safety Analysis**: Herb-drug interaction detection and contraindication checking
- **Prescription Analysis**: Extract entities from clinical prescriptions and provide Ayurvedic recommendations
- **Clinical Knowledge Query**: Access integrated knowledge base for disease-specific information

### Authentication & Authorization
- **JWT-based Authentication**: Secure token-based user authentication
- **Role-based Access Control**: Different access levels for general users and practitioners
- **Practitioner Verification**: Professional credential verification system
- **Admin Panel**: User management and credential approval workflow

### User Interfaces
- **General Users**: Access to medicine mapping and educational information
- **Qualified Practitioners**: Enhanced clinical tools including prescription analysis and safety assessments
- **Medicine Mapping**: Three-tab interface for medicine alternatives, disease recommendations, and symptom search
- **Model Comparison**: Instantaneous comparative analysis of treatment approaches
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
│   │   ├── prescription_routes.py # Prescription analysis API
│   │   ├── safety_analysis_routes.py   # Safety analysis API
│   │   ├── knowledge_routes.py    # Clinical knowledge API
│   │   ├── model_routes.py        # Model comparison API
│   │   └── admin_routes.py        # Admin management API
│   ├── services/                  # Core business logic services
│   │   ├── medicine_mapper.py     # Medicine mapping service
│   │   ├── auth_service.py        # Authentication service
│   │   ├── integrated_knowledge_system_optimized.py  # Knowledge system
│   │   ├── prescription_service_optimized.py         # Prescription analysis
│   │   └── safety_analyzer_optimized.py              # Safety analysis
│   ├── models/                    # Data models and schemas
│   │   └── user_models.py         # User authentication models
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
- `GET /api/models/comparison` - Get model performance comparison

### Admin (Administrators Only)
- `GET /api/admin/pending-verifications` - Get pending practitioner verifications
- `POST /api/admin/verify-practitioner` - Verify practitioner credentials

## Key Features in Detail

### Medicine Mapping
The medicine mapping system provides three main functionalities:

1. **Medicine Alternatives**: Find Ayurvedic alternatives for specific allopathic medicines
2. **Disease Recommendations**: Get comprehensive Ayurvedic treatment protocols for diseases
3. **Symptom Search**: Search for potential diseases and treatments based on symptoms

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
Comprehensive safety analysis includes:
- Herb-drug interaction detection
- Contraindication checking
- Dosage recommendations
- Precautions and warnings
- Risk assessment based on patient profile

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

The system integrates knowledge from:
- **Ayurgenix Dataset**: Primary dataset with 15+ diseases and Ayurvedic treatments
- **Traditional Texts**: Classical Ayurvedic literature
- **Modern Research**: Contemporary studies on herb-drug interactions
- **Clinical Guidelines**: Evidence-based treatment protocols

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
