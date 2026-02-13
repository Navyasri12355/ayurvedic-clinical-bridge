"""
Hybrid BiLSTM+Transformer models for prescription parsing and cross-domain mapping.

This module provides both BioBERT and BiLSTM-CRF models for clinical entity recognition,
with intelligent routing based on user requirements and query complexity.
"""

from .biobert_transformer import (
    BioBERTClinicalProcessor,
    BioBERTForClinicalNER,
    BioBERTConfig,
    ClinicalEntityType,
    TaskType,
    ClinicalPrediction,
    get_biobert_processor
)

from .bilstm_crf_model import (
    BiLSTMCRF,
    BiLSTMCRFConfig,
    BiLSTMCRFTokenizer,
    CRF,
    create_default_vocab
)

from .bilstm_crf_processor import (
    BiLSTMCRFClinicalProcessor,
    get_bilstm_crf_processor
)

from .model_manager import (
    UnifiedModelManager,
    ModelType,
    UserRole,
    ModelSelectionCriteria,
    ModelComparisonResult,
    get_unified_manager
)

__all__ = [
    # BioBERT components
    'BioBERTClinicalProcessor',
    'BioBERTForClinicalNER',
    'BioBERTConfig',
    'get_biobert_processor',
    
    # BiLSTM-CRF components
    'BiLSTMCRF',
    'BiLSTMCRFConfig',
    'BiLSTMCRFTokenizer',
    'BiLSTMCRFClinicalProcessor',
    'CRF',
    'create_default_vocab',
    'get_bilstm_crf_processor',
    
    # Unified management
    'UnifiedModelManager',
    'ModelType',
    'UserRole',
    'ModelSelectionCriteria',
    'ModelComparisonResult',
    'get_unified_manager',
    
    # Common types
    'ClinicalEntityType',
    'TaskType',
    'ClinicalPrediction'
]

# Optional imports to avoid dependency issues
# __all__ already contains core components, we will extend it


try:
    from .hybrid_ner import (
        HybridMedicalNER,
        PrescriptionParser,
        MedicalEntity,
        ParsedPrescription
    )
    __all__.extend([
        'HybridMedicalNER',
        'PrescriptionParser', 
        'MedicalEntity',
        'ParsedPrescription'
    ])
except ImportError:
    pass

try:
    from .cross_domain_mapper import (
        CrossDomainMapper,
        AyurvedicConcept,
        SemanticMapping,
        HybridEmbedding,
        ContrastiveLoss
    )
    __all__.extend([
        'CrossDomainMapper',
        'AyurvedicConcept',
        'SemanticMapping',
        'HybridEmbedding',
        'ContrastiveLoss'
    ])
except ImportError:
    pass

try:
    from .contrastive_learning import (
        AyurvedicDatasetLoader,
        ContrastiveTripletGenerator,
        ContrastiveDataset,
        ContrastiveLossFunction,
        CrossDomainEvaluator,
        TrainingPair,
        ContrastiveTriplet
    )
    __all__.extend([
        'AyurvedicDatasetLoader',
        'ContrastiveTripletGenerator',
        'ContrastiveDataset',
        'ContrastiveLossFunction',
        'CrossDomainEvaluator',
        'TrainingPair',
        'ContrastiveTriplet'
    ])
except ImportError:
    pass

try:
    from .contrastive_trainer import (
        ContrastiveTrainer,
        create_and_train_model
    )
    __all__.extend([
        'ContrastiveTrainer',
        'create_and_train_model'
    ])
except ImportError:
    pass

try:
    from .user_models import (
        User,
        UserCreate,
        UserLogin,
        UserResponse,
        UserRole,
        PractitionerCredentials,
        UserPreferences,
        Token,
        TokenData
    )
    __all__.extend([
        'User',
        'UserCreate',
        'UserLogin',
        'UserResponse',
        'UserRole',
        'PractitionerCredentials',
        'UserPreferences',
        'Token',
        'TokenData'
    ])
except ImportError:
    pass