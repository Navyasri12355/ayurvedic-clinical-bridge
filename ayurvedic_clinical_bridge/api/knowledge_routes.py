"""
Knowledge API Routes for Ayurvedic Clinical Bridge

This module provides API endpoints for the compiled knowledge system.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from ..services.knowledge_compiler import KnowledgeCompiler, CompiledKnowledge, KnowledgeCompilationResult
from ..services.integrated_knowledge_system_optimized import get_knowledge_system, KnowledgeQuery, KnowledgeResponse
from ..middleware.auth_middleware import get_current_user
from ..models.user_models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# Initialize services (commented out to avoid blocking)
# knowledge_compiler = KnowledgeCompiler()
# integrated_system = IntegratedKnowledgeSystem(knowledge_compiler=knowledge_compiler)

# Request/Response Models
class KnowledgeQueryRequest(BaseModel):
    query_text: str = Field(..., description="The query text to search for")
    query_type: str = Field(default="general", description="Type of query: general, disease, treatment, interaction")
    user_role: str = Field(default="general", description="User role: general or practitioner")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")

class KnowledgeQueryResponse(BaseModel):
    query_id: str
    concepts: List[Dict[str, Any]] = []
    cross_domain_mappings: List[Dict[str, Any]] = []
    confidence_score: float
    processing_time: float
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}

class CompilationRequest(BaseModel):
    force_recompile: bool = Field(default=False, description="Force recompilation even if recent compilation exists")

class CompilationResponse(BaseModel):
    success: bool
    total_concepts: int
    total_relationships: int
    compilation_time: float
    output_file: str
    file_size_mb: float
    errors: List[str] = []
    warnings: List[str] = []

class SystemStatsResponse(BaseModel):
    total_concepts: int
    concept_types: Dict[str, int]
    source_distribution: Dict[str, int]
    has_cross_domain_mapper: bool
    max_results: int
    last_compilation: Optional[str] = None

@router.get("/health")
async def knowledge_health():
    """Simple health check for knowledge system."""
    try:
        # Quick check without loading full knowledge base
        return {
            "status": "healthy",
            "knowledge_system": "operational",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/query", response_model=KnowledgeQueryResponse)
async def query_knowledge(
    request: KnowledgeQueryRequest,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Query the compiled knowledge base for relevant information.
    
    This endpoint searches the compiled knowledge base and returns relevant concepts
    and cross-domain mappings based on the query.
    """
    try:
        # Get the optimized knowledge system
        knowledge_system = get_knowledge_system()
        
        # Create knowledge query
        query = KnowledgeQuery(
            query_text=request.query_text,
            query_type=request.query_type,
            user_role=request.user_role,
            filters=request.filters
        )
        
        # Process the query
        response = knowledge_system.query_knowledge(query)
        
        # Convert to API response format
        api_response = KnowledgeQueryResponse(
            query_id=response.query_id,
            concepts=response.concepts,
            cross_domain_mappings=response.cross_domain_mappings,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            warnings=response.warnings,
            metadata=response.metadata
        )
        
        logger.info(f"Knowledge query processed: {request.query_text[:50]}... -> {len(response.concepts)} concepts, {len(response.cross_domain_mappings)} mappings")
        return api_response
        
    except Exception as e:
        logger.error(f"Knowledge query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge query failed: {str(e)}"
        )

@router.get("/concept/{concept_id}")
async def get_concept(
    concept_id: str,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Get a specific concept by ID.
    """
    try:
        knowledge_system = get_knowledge_system()
        concept = knowledge_system.get_concept_by_id(concept_id)
        
        if not concept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept not found: {concept_id}"
            )
        
        return concept
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get concept {concept_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get concept: {str(e)}"
        )

@router.get("/concept/{concept_id}/related")
async def get_related_concepts(
    concept_id: str,
    relation_type: Optional[str] = None,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Get concepts related to a specific concept.
    """
    try:
        related_concepts = integrated_system.get_related_concepts(concept_id, relation_type)
        return {
            "concept_id": concept_id,
            "relation_type": relation_type,
            "related_concepts": [concept.__dict__ for concept in related_concepts]
        }
        
    except Exception as e:
        logger.error(f"Failed to get related concepts for {concept_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get related concepts: {str(e)}"
        )

@router.post("/compile", response_model=CompilationResponse)
async def compile_knowledge(
    request: CompilationRequest,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Compile all knowledge sources into a single JSON file.
    
    This endpoint triggers the knowledge compilation process, which combines
    all available knowledge sources into a unified format for efficient retrieval.
    """
    try:
        # Check if user has permission (could be restricted to admin users)
        if current_user and hasattr(current_user, 'role') and current_user.role not in ['admin', 'practitioner']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to compile knowledge base"
            )
        
        logger.info("Starting knowledge compilation...")
        result = knowledge_compiler.compile_all_knowledge_sources()
        
        if result.success:
            # Refresh the integrated system with new compilation
            integrated_system.refresh_knowledge_base()
            logger.info("Knowledge compilation completed successfully")
        
        return CompilationResponse(
            success=result.success,
            total_concepts=result.total_concepts,
            total_relationships=result.total_relationships,
            compilation_time=result.compilation_time,
            output_file=result.output_file,
            file_size_mb=result.file_size_mb,
            errors=result.errors,
            warnings=result.warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge compilation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge compilation failed: {str(e)}"
        )

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Get system statistics and information about the knowledge base.
    """
    try:
        knowledge_system = get_knowledge_system()
        stats = knowledge_system.get_system_stats()
        
        return SystemStatsResponse(
            total_concepts=stats['total_concepts'],
            concept_types=stats['concept_types'],
            source_distribution=stats['source_distribution'],
            has_cross_domain_mapper=stats['has_cross_domain_mapper'],
            max_results=stats['max_results'],
            last_compilation=datetime.now().isoformat()  # Could be stored in metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system stats: {str(e)}"
        )

@router.post("/refresh")
async def refresh_knowledge_base(
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Refresh the knowledge base from the latest compiled data.
    """
    try:
        # Check permissions
        if current_user and hasattr(current_user, 'role') and current_user.role not in ['admin', 'practitioner']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to refresh knowledge base"
            )
        
        knowledge_system = get_knowledge_system()
        success = knowledge_system.refresh_knowledge_base()
        
        if success:
            return {"success": True, "message": "Knowledge base refreshed successfully"}
        else:
            return {"success": False, "message": "Failed to refresh knowledge base"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh knowledge base: {str(e)}"
        )

@router.get("/search")
async def search_concepts(
    q: str,
    concept_type: Optional[str] = None,
    limit: int = 10,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Search for concepts in the knowledge base.
    """
    try:
        if not q or not q.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query parameter 'q' is required"
            )
        
        concepts = knowledge_compiler.search_compiled_knowledge(
            query=q.strip(),
            concept_type=concept_type,
            limit=min(limit, 50)  # Cap at 50 results
        )
        
        return {
            "query": q,
            "concept_type": concept_type,
            "limit": limit,
            "results": [concept.__dict__ for concept in concepts]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed for query '{q}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
