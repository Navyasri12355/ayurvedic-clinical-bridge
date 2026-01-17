"""
Knowledge Compilation System for Ayurvedic Clinical Bridge

This module replaces the RAG system with a unified knowledge compilation approach.
All knowledge sources are compiled into a single JSON file for efficient learning and retrieval.
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import numpy as np

from ..data.dataset_integration import AyurvedicDatasetIntegrator, ProcessedEntity

logger = logging.getLogger(__name__)

@dataclass
class CompiledKnowledge:
    """Represents compiled knowledge from multiple sources"""
    id: str
    concept_name: str
    concept_type: str  # 'disease', 'herb', 'symptom', 'treatment', 'interaction'
    biomedical_terms: List[str] = field(default_factory=list)
    ayurvedic_terms: List[str] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    last_updated: str = ""

@dataclass
class KnowledgeCompilationResult:
    """Result of knowledge compilation process"""
    success: bool
    total_concepts: int = 0
    total_relationships: int = 0
    compilation_time: float = 0.0
    output_file: str = ""
    file_size_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class KnowledgeCompiler:
    """
    Compiles multiple knowledge sources into a single JSON file for learning and retrieval.
    
    This replaces the RAG system with a simpler approach where all knowledge is
    pre-compiled and stored in a structured JSON format that can be efficiently
    loaded and searched during inference.
    """
    
    def __init__(self, output_dir: str = "data/compiled_knowledge"):
        """
        Initialize the knowledge compiler.
        
        Args:
            output_dir: Directory to store compiled knowledge files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.compiled_knowledge: Dict[str, CompiledKnowledge] = {}
        self.concept_index: Dict[str, Set[str]] = {}  # For fast lookup
        self.relationship_index: Dict[str, List[str]] = {}
        
        # Initialize dataset integrator
        self.dataset_integrator = AyurvedicDatasetIntegrator()
        
    def compile_all_knowledge_sources(self) -> KnowledgeCompilationResult:
        """
        Compile all available knowledge sources into a single JSON file.
        
        Returns:
            KnowledgeCompilationResult with compilation statistics
        """
        start_time = datetime.now()
        result = KnowledgeCompilationResult(success=False)
        
        try:
            logger.info("Starting knowledge compilation process...")
            
            # 1. Load and process all datasets
            self._compile_dataset_knowledge()
            
            # 2. Create cross-references and relationships
            self._create_cross_references()
            
            # 4. Generate output JSON file
            output_file = self._generate_compiled_json()
            
            # Calculate statistics
            end_time = datetime.now()
            compilation_time = (end_time - start_time).total_seconds()
            
            result.success = True
            result.total_concepts = len(self.compiled_knowledge)
            result.total_relationships = sum(len(k.relationships) for k in self.compiled_knowledge.values())
            result.compilation_time = compilation_time
            result.output_file = output_file
            result.file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
            
            logger.info(f"Knowledge compilation completed successfully:")
            logger.info(f"  - Total concepts: {result.total_concepts}")
            logger.info(f"  - Total relationships: {result.total_relationships}")
            logger.info(f"  - Compilation time: {compilation_time:.2f}s")
            logger.info(f"  - Output file: {output_file}")
            logger.info(f"  - File size: {result.file_size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Knowledge compilation failed: {str(e)}")
            result.errors.append(str(e))
            
        return result
    
    def _compile_dataset_knowledge(self):
        """Compile knowledge from all integrated datasets."""
        logger.info("Compiling dataset knowledge...")
        
        try:
            # Load all datasets
            datasets_result = self.dataset_integrator.integrate_all_datasets()
            
            if not datasets_result:
                logger.warning("No datasets were loaded")
                return
                
            # Process each dataset
            for dataset_name, metadata in datasets_result.items():
                entities = self.dataset_integrator.processed_entities.get(dataset_name, [])
                logger.info(f"Processing {len(entities)} entities from {dataset_name}")
                
                for entity in entities:
                    self._add_entity_to_knowledge(entity, dataset_name)
                    
        except Exception as e:
            logger.error(f"Failed to compile dataset knowledge: {str(e)}")
            # Don't raise - continue with other knowledge sources
    
    def _add_entity_to_knowledge(self, entity: ProcessedEntity, source: str):
        """Add a processed entity to compiled knowledge."""
        concept_id = self._generate_concept_id(entity.name, entity.entity_type)
        
        if concept_id in self.compiled_knowledge:
            # Merge with existing concept
            existing = self.compiled_knowledge[concept_id]
            existing.descriptions.append(entity.description)
            existing.properties.update(entity.properties)
            existing.sources.append(source)
            # Handle synonyms safely
            synonyms = getattr(entity, 'synonyms', []) or []
            existing.biomedical_terms.extend(synonyms) if entity.entity_type in ['disease', 'symptom', 'drug'] else existing.ayurvedic_terms.extend(synonyms)
        else:
            # Create new concept
            compiled = CompiledKnowledge(
                id=concept_id,
                concept_name=entity.name,
                concept_type=entity.entity_type,
                descriptions=[entity.description] if entity.description else [],
                properties=entity.properties.copy(),
                sources=[source],
                confidence_score=entity.confidence_score,
                last_updated=datetime.now().isoformat()
            )
            
            # Categorize terms
            if entity.entity_type in ['disease', 'symptom', 'drug']:
                compiled.biomedical_terms.append(entity.name)
                if hasattr(entity, 'synonyms') and entity.synonyms:
                    compiled.biomedical_terms.extend(entity.synonyms)
            elif entity.entity_type in ['herb', 'remedy', 'treatment']:
                compiled.ayurvedic_terms.append(entity.name)
                if hasattr(entity, 'synonyms') and entity.synonyms:
                    compiled.ayurvedic_terms.extend(entity.synonyms)
            
            self.compiled_knowledge[concept_id] = compiled
            
        # Update indexes
        synonyms = getattr(entity, 'synonyms', []) or []
        self._update_concept_index(concept_id, entity.name, synonyms)
    
    def _create_cross_references(self):
        """Create cross-references between biomedical and Ayurvedic concepts."""
        logger.info("Creating cross-references...")
        
        # Simple similarity-based cross-referencing
        biomedical_concepts = [k for k in self.compiled_knowledge.values() 
                             if k.concept_type in ['disease', 'symptom', 'drug']]
        ayurvedic_concepts = [k for k in self.compiled_knowledge.values() 
                            if k.concept_type in ['herb', 'remedy', 'treatment']]
        
        for bio_concept in biomedical_concepts:
            for ayu_concept in ayurvedic_concepts:
                similarity = self._calculate_concept_similarity(bio_concept, ayu_concept)
                if similarity > 0.3:  # Threshold for cross-reference
                    cross_ref = {
                        'type': 'CROSS_DOMAIN_MAPPING',
                        'target_concept': ayu_concept.concept_name,
                        'target_id': ayu_concept.id,
                        'similarity_score': similarity,
                        'source': 'cross_reference_generation'
                    }
                    bio_concept.relationships.append(cross_ref)
    
    def _calculate_concept_similarity(self, concept1: CompiledKnowledge, concept2: CompiledKnowledge) -> float:
        """Calculate similarity between two concepts based on descriptions and properties."""
        # Simple keyword-based similarity
        desc1 = ' '.join(concept1.descriptions).lower()
        desc2 = ' '.join(concept2.descriptions).lower()
        
        if not desc1 or not desc2:
            return 0.0
            
        words1 = set(desc1.split())
        words2 = set(desc2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_compiled_json(self) -> str:
        """Generate the final compiled JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"compiled_knowledge_{timestamp}.json"
        
        # Convert to serializable format
        compiled_data = {
            'metadata': {
                'compilation_date': datetime.now().isoformat(),
                'total_concepts': len(self.compiled_knowledge),
                'total_relationships': sum(len(k.relationships) for k in self.compiled_knowledge.values()),
                'version': '1.0'
            },
            'concepts': {k: asdict(v) for k, v in self.compiled_knowledge.items()},
            'indexes': {
                'concept_index': {k: list(v) for k, v in self.concept_index.items()},
                'relationship_index': self.relationship_index
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(compiled_data, f, indent=2, ensure_ascii=False)
            
        # Also create a "latest" copy
        latest_file = self.output_dir / "compiled_knowledge_latest.json"
        try:
            if latest_file.exists():
                latest_file.unlink()
            # Copy instead of symlink for Windows compatibility
            import shutil
            shutil.copy2(output_file, latest_file)
        except Exception as e:
            logger.warning(f"Could not create latest file: {str(e)}")
        
        return str(output_file)
    
    def _generate_concept_id(self, name: str, concept_type: str) -> str:
        """Generate a unique concept ID."""
        content = f"{concept_type}:{name.lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _find_concept_id(self, name: str) -> Optional[str]:
        """Find concept ID by name."""
        name_lower = name.lower()
        for concept_id, concept in self.compiled_knowledge.items():
            if (concept.concept_name.lower() == name_lower or 
                name_lower in [term.lower() for term in concept.biomedical_terms + concept.ayurvedic_terms]):
                return concept_id
        return None
    
    def _update_concept_index(self, concept_id: str, name: str, synonyms: List[str]):
        """Update the concept index for fast lookup."""
        terms = [name.lower()] + [s.lower() for s in synonyms]
        for term in terms:
            if term not in self.concept_index:
                self.concept_index[term] = set()
            self.concept_index[term].add(concept_id)
    
    def load_compiled_knowledge(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load compiled knowledge from JSON file.
        
        Args:
            file_path: Path to compiled knowledge file. If None, loads latest.
            
        Returns:
            Loaded knowledge data
        """
        if file_path is None:
            file_path = self.output_dir / "compiled_knowledge_latest.json"
            
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Compiled knowledge file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def search_compiled_knowledge(self, query: str, concept_type: Optional[str] = None, 
                                limit: int = 10) -> List[CompiledKnowledge]:
        """
        Search compiled knowledge for relevant concepts.
        
        Args:
            query: Search query
            concept_type: Filter by concept type
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        query_lower = query.lower()
        results = []
        
        for concept in self.compiled_knowledge.values():
            if concept_type and concept.concept_type != concept_type:
                continue
                
            # Check if query matches name, synonyms, or descriptions
            if (query_lower in concept.concept_name.lower() or
                any(query_lower in term.lower() for term in concept.biomedical_terms + concept.ayurvedic_terms) or
                any(query_lower in desc.lower() for desc in concept.descriptions)):
                results.append(concept)
                
        # Sort by relevance (simple name match first)
        results.sort(key=lambda x: query_lower in x.concept_name.lower(), reverse=True)
        
        return results[:limit]