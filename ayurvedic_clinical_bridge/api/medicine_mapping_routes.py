"""
Medicine Mapping API Routes for Ayurvedic Clinical Bridge

This module provides API endpoints for mapping between allopathic and ayurvedic medicines.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from ..middleware.auth_middleware import get_current_user, require_general_user_or_practitioner
from ..models.user_models import User
from ..services.medicine_mapper import get_medicine_mapper, MedicineMapping, AyurvedicRecommendation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/medicine-mapping", tags=["medicine-mapping"])

# Request/Response Models
class MedicineMappingRequest(BaseModel):
    allopathic_medicine: str = Field(..., description="Name of the allopathic medicine")
    disease: Optional[str] = Field(None, description="Disease context for better mapping")

class MedicineMappingResponse(BaseModel):
    allopathic_medicine: str
    ayurvedic_alternatives: List[str]
    disease: str
    dosage: str
    formulation: str
    dosha: str
    constitution: str
    confidence_score: float
    safety_notes: str
    contraindications: str
    interaction_warnings: List[str] = []

class DiseaseRecommendationRequest(BaseModel):
    disease: str = Field(..., description="Name of the disease")

class AyurvedicRecommendationResponse(BaseModel):
    herb_name: str
    dosage: str
    formulation: str
    preparation_method: str
    timing: str
    duration: str
    precautions: str

class SymptomSearchRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptoms to search for")

class SymptomSearchResponse(BaseModel):
    disease: str
    symptoms: str
    ayurvedic_herbs: str
    formulation: str
    dosha: str
    constitution: str
    diet_recommendations: str

class InteractionCheckRequest(BaseModel):
    allopathic_medicine: str = Field(..., description="Allopathic medicine name")
    ayurvedic_herbs: List[str] = Field(..., description="List of ayurvedic herbs")

class InteractionCheckResponse(BaseModel):
    allopathic_medicine: str
    ayurvedic_herbs: List[str]
    warnings: List[str]
    safe_combinations: List[str]

@router.post("/find-alternative", response_model=MedicineMappingResponse)
async def find_ayurvedic_alternative(
    request: MedicineMappingRequest,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Find ayurvedic alternatives for a given allopathic medicine.
    
    This endpoint maps allopathic medicines to their ayurvedic equivalents
    based on the ayurgenix dataset and traditional knowledge.
    """
    try:
        mapper = get_medicine_mapper()
        mapping = mapper.find_ayurvedic_alternative(
            request.allopathic_medicine, 
            request.disease
        )
        
        if not mapping:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No ayurvedic alternative found for {request.allopathic_medicine}"
            )
        
        # Get interaction warnings
        warnings = mapper.get_interaction_warnings(
            request.allopathic_medicine,
            mapping.ayurvedic_alternatives
        )
        
        return MedicineMappingResponse(
            allopathic_medicine=mapping.allopathic_medicine,
            ayurvedic_alternatives=mapping.ayurvedic_alternatives,
            disease=mapping.disease,
            dosage=mapping.dosage,
            formulation=mapping.formulation,
            dosha=mapping.dosha,
            constitution=mapping.constitution,
            confidence_score=mapping.confidence_score,
            safety_notes=mapping.safety_notes,
            contraindications=mapping.contraindications,
            interaction_warnings=warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding ayurvedic alternative: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find ayurvedic alternative"
        )

@router.post("/disease-recommendations", response_model=List[AyurvedicRecommendationResponse])
async def get_disease_recommendations(
    request: DiseaseRecommendationRequest,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Get ayurvedic medicine recommendations based on disease.
    
    This endpoint provides comprehensive ayurvedic treatment recommendations
    for specific diseases based on traditional knowledge and the ayurgenix dataset.
    """
    try:
        mapper = get_medicine_mapper()
        recommendations = mapper.get_disease_based_recommendations(request.disease)
        
        if not recommendations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No ayurvedic recommendations found for {request.disease}"
            )
        
        return [
            AyurvedicRecommendationResponse(
                herb_name=rec.herb_name,
                dosage=rec.dosage,
                formulation=rec.formulation,
                preparation_method=rec.preparation_method,
                timing=rec.timing,
                duration=rec.duration,
                precautions=rec.precautions
            )
            for rec in recommendations
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting disease recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get disease recommendations"
        )

@router.post("/search-by-symptoms", response_model=List[SymptomSearchResponse])
async def search_by_symptoms(
    request: SymptomSearchRequest,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Search for ayurvedic recommendations based on symptoms.
    
    This endpoint allows users to search for potential diseases and
    their ayurvedic treatments based on reported symptoms.
    """
    try:
        mapper = get_medicine_mapper()
        results = mapper.search_by_symptoms(request.symptoms)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No matching diseases found for the provided symptoms"
            )
        
        return [
            SymptomSearchResponse(
                disease=result["disease"],
                symptoms=result["symptoms"],
                ayurvedic_herbs=result["ayurvedic_herbs"],
                formulation=result["formulation"],
                dosha=result["dosha"],
                constitution=result["constitution"],
                diet_recommendations=result["diet_recommendations"]
            )
            for result in results
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching by symptoms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search by symptoms"
        )

@router.post("/check-interactions", response_model=InteractionCheckResponse)
async def check_medicine_interactions(
    request: InteractionCheckRequest,
    current_user: User = Depends(require_general_user_or_practitioner)
):
    """
    Check for potential interactions between allopathic medicines and ayurvedic herbs.
    
    This endpoint provides safety information about combining allopathic
    medicines with ayurvedic herbs.
    """
    try:
        mapper = get_medicine_mapper()
        warnings = mapper.get_interaction_warnings(
            request.allopathic_medicine,
            request.ayurvedic_herbs
        )
        
        # Determine safe combinations (herbs without warnings)
        herbs_with_warnings = set()
        for warning in warnings:
            for herb in request.ayurvedic_herbs:
                if herb.lower() in warning.lower():
                    herbs_with_warnings.add(herb)
        
        safe_combinations = [
            herb for herb in request.ayurvedic_herbs 
            if herb not in herbs_with_warnings
        ]
        
        return InteractionCheckResponse(
            allopathic_medicine=request.allopathic_medicine,
            ayurvedic_herbs=request.ayurvedic_herbs,
            warnings=warnings,
            safe_combinations=safe_combinations
        )
        
    except Exception as e:
        logger.error(f"Error checking interactions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check medicine interactions"
        )

@router.get("/popular-mappings")
async def get_popular_mappings(
    current_user: User = Depends(get_current_user)
):
    """
    Get a list of popular allopathic to ayurvedic medicine mappings.
    
    This endpoint provides commonly requested medicine mappings for
    educational purposes.
    """
    try:
        popular_mappings = [
            {
                "allopathic": "Metformin",
                "ayurvedic": "Jamun, Gudmar, Fenugreek",
                "condition": "Diabetes",
                "confidence": "High"
            },
            {
                "allopathic": "Ibuprofen",
                "ayurvedic": "Turmeric, Boswellia, Ginger",
                "condition": "Pain & Inflammation",
                "confidence": "High"
            },
            {
                "allopathic": "Omeprazole",
                "ayurvedic": "Amla, Aloe Vera, Licorice",
                "condition": "Acidity",
                "confidence": "High"
            },
            {
                "allopathic": "Amlodipine",
                "ayurvedic": "Arjuna, Ashwagandha, Garlic",
                "condition": "Hypertension",
                "confidence": "High"
            },
            {
                "allopathic": "Lorazepam",
                "ayurvedic": "Ashwagandha, Brahmi, Jatamansi",
                "condition": "Anxiety",
                "confidence": "Medium"
            }
        ]
        
        return {
            "popular_mappings": popular_mappings,
            "disclaimer": "These mappings are for educational purposes only. Always consult with qualified healthcare professionals before making any changes to your medication regimen."
        }
        
    except Exception as e:
        logger.error(f"Error getting popular mappings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get popular mappings"
        )

@router.get("/herbs-database")
async def get_herbs_database(
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get information about ayurvedic herbs from the database.
    
    This endpoint provides a searchable database of ayurvedic herbs
    with their properties and uses.
    """
    try:
        # Common ayurvedic herbs database
        herbs_db = [
            {
                "name": "Ashwagandha",
                "scientific_name": "Withania somnifera",
                "properties": "Adaptogenic, Anti-stress, Immunomodulatory",
                "uses": "Anxiety, Insomnia, Fatigue, Immune support",
                "dosha": "Vata-Pitta",
                "dosage": "3-6g daily",
                "precautions": "Avoid in autoimmune conditions"
            },
            {
                "name": "Turmeric",
                "scientific_name": "Curcuma longa",
                "properties": "Anti-inflammatory, Antioxidant, Hepatoprotective",
                "uses": "Inflammation, Arthritis, Digestive issues",
                "dosha": "Pitta-Kapha",
                "dosage": "1-3g daily",
                "precautions": "May interact with blood thinners"
            },
            {
                "name": "Tulsi",
                "scientific_name": "Ocimum sanctum",
                "properties": "Adaptogenic, Antimicrobial, Respiratory tonic",
                "uses": "Respiratory issues, Stress, Immunity",
                "dosha": "Kapha-Vata",
                "dosage": "5-10 leaves daily",
                "precautions": "Generally safe"
            },
            {
                "name": "Brahmi",
                "scientific_name": "Bacopa monnieri",
                "properties": "Nootropic, Memory enhancer, Nervine tonic",
                "uses": "Memory, Concentration, Anxiety",
                "dosha": "Vata-Pitta",
                "dosage": "2-4g daily",
                "precautions": "May cause drowsiness initially"
            },
            {
                "name": "Arjuna",
                "scientific_name": "Terminalia arjuna",
                "properties": "Cardiotonic, Hypotensive, Antioxidant",
                "uses": "Heart health, Hypertension, Cholesterol",
                "dosha": "Pitta-Kapha",
                "dosage": "3-6g daily",
                "precautions": "Monitor blood pressure"
            }
        ]
        
        # Filter by search term if provided
        if search:
            search_lower = search.lower()
            herbs_db = [
                herb for herb in herbs_db
                if search_lower in herb["name"].lower() or 
                   search_lower in herb["uses"].lower() or
                   search_lower in herb["properties"].lower()
            ]
        
        return {
            "herbs": herbs_db,
            "total_count": len(herbs_db),
            "search_term": search
        }
        
    except Exception as e:
        logger.error(f"Error getting herbs database: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get herbs database"
        )