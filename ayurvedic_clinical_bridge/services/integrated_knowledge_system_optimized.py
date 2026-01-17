"""
Optimized Integrated Knowledge System for faster loading and real-time queries.

This module provides an optimized version of the knowledge system that:
1. Loads knowledge lazily
2. Uses efficient indexing
3. Provides real query processing
4. Maintains good performance
"""

import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from .confidence_scorer import get_confidence_scorer
from .cross_domain_mapper import get_cross_domain_mapper, MappingQuery

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeQuery:
    """Knowledge query request."""
    query_text: str
    query_type: str = "general"
    user_role: str = "general"
    filters: Dict[str, Any] = None

@dataclass
class KnowledgeResponse:
    """Knowledge query response."""
    query_id: str
    concepts: List[Dict[str, Any]]
    cross_domain_mappings: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    warnings: List[str]
    metadata: Dict[str, Any]

class OptimizedIntegratedKnowledgeSystem:
    """Optimized knowledge system with lazy loading and efficient querying."""
    
    def __init__(self, knowledge_file_path: Optional[str] = None):
        """Initialize with lazy loading."""
        self.knowledge_file_path = knowledge_file_path or self._get_latest_knowledge_file()
        self._knowledge_data = None
        self._concept_index = None
        self._term_index = None
        self._loaded = False
        self.confidence_scorer = get_confidence_scorer()
        self.cross_domain_mapper = get_cross_domain_mapper()
        
        logger.info(f"Initialized OptimizedIntegratedKnowledgeSystem with file: {self.knowledge_file_path}")
    
    def _get_latest_knowledge_file(self) -> str:
        """Get the latest compiled knowledge file."""
        base_dir = Path(__file__).parent.parent.parent
        knowledge_dir = base_dir / "data" / "compiled_knowledge"
        
        latest_file = knowledge_dir / "compiled_knowledge_latest.json"
        if latest_file.exists():
            return str(latest_file)
        
        # Fallback to most recent timestamped file
        json_files = list(knowledge_dir.glob("compiled_knowledge_*.json"))
        if json_files:
            latest = max(json_files, key=lambda f: f.stat().st_mtime)
            return str(latest)
        
        raise FileNotFoundError("No compiled knowledge file found")
    
    def _ensure_loaded(self):
        """Ensure knowledge data is loaded (lazy loading)."""
        if self._loaded:
            return
        
        start_time = time.time()
        logger.info("Loading compiled knowledge data...")
        
        try:
            with open(self.knowledge_file_path, 'r', encoding='utf-8') as f:
                self._knowledge_data = json.load(f)
            
            # Build efficient indexes
            self._build_indexes()
            self._loaded = True
            
            load_time = time.time() - start_time
            concept_count = len(self._knowledge_data.get('concepts', []))
            logger.info(f"Loaded {concept_count} concepts in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge data: {str(e)}")
            # Create minimal fallback data
            self._knowledge_data = {"concepts": [], "relationships": []}
            self._build_indexes()
            self._loaded = True
    
    def _build_indexes(self):
        """Build efficient search indexes."""
        self._concept_index = {}
        self._term_index = {}
        
        # Handle different data formats
        concepts_data = self._knowledge_data.get('concepts', {})
        
        # If concepts is a dict (keyed by ID), convert to list
        if isinstance(concepts_data, dict):
            concepts = list(concepts_data.values())
        else:
            concepts = concepts_data
        
        for i, concept in enumerate(concepts):
            concept_id = concept.get('id', f'concept_{i}')
            self._concept_index[concept_id] = concept
            
            # Index searchable terms
            terms = []
            
            # Add concept name
            concept_name = concept.get('concept_name', concept.get('name', ''))
            if concept_name:
                terms.append(concept_name.lower())
            
            # Add descriptions
            descriptions = concept.get('descriptions', [])
            if descriptions:
                terms.extend([desc.lower() for desc in descriptions[:5]])  # Limit to first 5
            
            # Add Ayurvedic terms
            ayurvedic_terms = concept.get('ayurvedic_terms', [])
            if ayurvedic_terms:
                terms.extend([term.lower() for term in ayurvedic_terms])
            
            # Add biomedical terms
            biomedical_terms = concept.get('biomedical_terms', [])
            if biomedical_terms:
                terms.extend([term.lower() for term in biomedical_terms])
            
            # Add English terms if available
            english_terms = concept.get('english_terms', [])
            if english_terms:
                terms.extend([term.lower() for term in english_terms])
            
            # Index terms
            for term in terms:
                if term and len(term) > 2:  # Skip very short terms
                    words = term.split()
                    for word in words:
                        if len(word) > 2:
                            if word not in self._term_index:
                                self._term_index[word] = []
                            self._term_index[word].append((concept_id, concept))
    
    def query_knowledge(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """Process a knowledge query."""
        start_time = time.time()
        self._ensure_loaded()
        
        query_id = f"query_{int(time.time() * 1000)}"
        
        # Search for relevant concepts
        concepts, content_type = self._search_concepts(query.query_text, query.query_type, query.user_role)
        
        # Check for cross-domain mapping queries
        cross_domain_mappings = self._handle_cross_domain_queries(query.query_text)
        
        processing_time = time.time() - start_time
        
        # Calculate dynamic confidence score
        confidence_score = self.confidence_scorer.calculate_confidence(
            query_text=query.query_text,
            concepts=concepts,
            content_type=content_type,
            processing_time=processing_time,
            metadata={
                "query_type": query.query_type,
                "user_role": query.user_role,
                "total_concepts_searched": len(self._knowledge_data.get('concepts', {})),
                "results_returned": len(concepts),
                "cross_domain_mappings_found": len(cross_domain_mappings)
            }
        )
        
        # Boost confidence if we have cross-domain mappings but no regular concepts
        if cross_domain_mappings and len(concepts) == 0:
            # Use the average confidence of cross-domain mappings
            avg_mapping_confidence = sum(m.get('confidence_score', 0) for m in cross_domain_mappings) / len(cross_domain_mappings)
            confidence_score = max(confidence_score, avg_mapping_confidence * 0.8)  # 80% of mapping confidence
        
        # Generate warnings based on user role
        warnings = self._generate_warnings(query.user_role, concepts)
        
        return KnowledgeResponse(
            query_id=query_id,
            concepts=concepts,
            cross_domain_mappings=cross_domain_mappings,
            confidence_score=confidence_score,
            processing_time=processing_time,
            warnings=warnings,
            metadata={
                "query_type": query.query_type,
                "user_role": query.user_role,
                "total_concepts_searched": len(self._knowledge_data.get('concepts', {})),
                "results_returned": len(concepts),
                "content_type": content_type,
                "confidence_explanation": self.confidence_scorer.get_confidence_explanation(confidence_score),
                "cross_domain_mappings_found": len(cross_domain_mappings)
            }
        )
    
    def _search_concepts(self, query_text: str, query_type: str, user_role: str) -> Tuple[List[Dict[str, Any]], str]:
        """Search for concepts matching the query. Returns (concepts, content_type)."""
        query_lower = query_text.lower()
        query_words = query_lower.split()
        
        # Check if this is a cross-domain query first
        cross_domain_patterns = [
            "ayurvedic equivalent of",
            "natural alternative to",
            "herbal substitute for",
            "ayurvedic medicine for",
            "allopathic equivalent of",
            "modern medicine for",
            "western medicine for"
        ]
        
        is_cross_domain_query = any(pattern in query_lower for pattern in cross_domain_patterns)
        
        # For cross-domain queries, prioritize cross-domain results and return minimal regular concepts
        if is_cross_domain_query:
            # Extract the medicine name from cross-domain query
            medicine_name = query_lower
            for pattern in cross_domain_patterns:
                medicine_name = medicine_name.replace(pattern, '').strip()
            
            # Search for concepts related to the specific medicine
            medicine_concepts = self._search_medicine_specific_concepts(medicine_name, user_role)
            
            if medicine_concepts:
                return medicine_concepts, "cross_domain_focused"
            else:
                # Fallback to general search but limit results
                return self._search_general_concepts(query_words, query_type, user_role, limit=3), "cross_domain_fallback"
        
        # Handle specific fundamental queries first
        fundamental_response = self._handle_fundamental_queries(query_lower, user_role)
        if fundamental_response:
            # Determine content type based on query
            if any(word in query_lower for word in ["dosha", "doshas", "panchakarma", "what is ayurveda"]):
                content_type = "curated_fundamental"
            elif any(word in query_lower for word in ["turmeric", "ginger", "ashwagandha", "neem", "brahmi"]):
                content_type = "curated_herb"
            else:
                content_type = "curated_health"
            return fundamental_response, content_type
        
        # Regular search for non-cross-domain queries
        return self._search_general_concepts(query_words, query_type, user_role), "database_search"
    
    def _search_medicine_specific_concepts(self, medicine_name: str, user_role: str) -> List[Dict[str, Any]]:
        """Search for concepts specifically related to a medicine name."""
        medicine_words = medicine_name.split()
        concept_scores = {}
        
        # Focus on the medicine name specifically
        for word in medicine_words:
            if len(word) > 2 and word in self._term_index:
                for concept_id, concept in self._term_index[word]:
                    concept_name = concept.get('concept_name', '').lower()
                    
                    # Only include concepts that are actually about the medicine
                    if word in concept_name or any(med_word in concept_name for med_word in medicine_words):
                        if concept_id not in concept_scores:
                            concept_scores[concept_id] = {"concept": concept, "score": 0, "relevance": 0}
                        concept_scores[concept_id]["score"] += 1
                        concept_scores[concept_id]["relevance"] += 2  # Higher relevance for medicine-specific
        
        # Filter and format results
        filtered_concepts = {}
        for concept_id, data in concept_scores.items():
            concept = data["concept"]
            concept_name = concept.get('concept_name', '')
            
            # Skip irrelevant or fragmented results
            if len(concept_name) < 10:
                continue
                
            # Skip QA fragments
            if concept_name.startswith(('A1:', 'A2:', 'A3:', 'Q1:', 'Q2:', 'Q3:')):
                continue
            
            # Skip generic remedies that don't mention the medicine
            if not any(word in concept_name.lower() for word in medicine_name.split()):
                continue
                
            filtered_concepts[concept_id] = data
        
        # Sort by relevance and score
        sorted_concepts = sorted(
            filtered_concepts.values(),
            key=lambda x: (x["relevance"], x["score"]),
            reverse=True
        )
        
        # Format results
        results = []
        for item in sorted_concepts[:5]:  # Limit to top 5 medicine-specific results
            concept = item["concept"]
            formatted_concept = self._format_concept(concept, user_role)
            results.append(formatted_concept)
        
        return results
    
    def _search_general_concepts(self, query_words: List[str], query_type: str, user_role: str, limit: int = 10) -> List[Dict[str, Any]]:
        """General concept search with improved filtering."""
        # Find matching concepts
        concept_scores = {}
        
        # Skip very common words that don't add value
        skip_words = {'of', 'the', 'and', 'or', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'is', 'are', 'was', 'were'}
        meaningful_words = [word for word in query_words if word not in skip_words and len(word) > 2]
        
        for word in meaningful_words:
            if word in self._term_index:
                # Limit the number of concepts we consider for very common words
                concepts_to_consider = self._term_index[word]
                if len(concepts_to_consider) > 1000:  # If too many matches, be more selective
                    concepts_to_consider = concepts_to_consider[:1000]
                
                for concept_id, concept in concepts_to_consider:
                    if concept_id not in concept_scores:
                        concept_scores[concept_id] = {"concept": concept, "score": 0, "relevance": 0}
                    concept_scores[concept_id]["score"] += 1
                    
                    # Boost relevance for exact matches in concept name
                    concept_name = concept.get('concept_name', '').lower()
                    if word in concept_name:
                        concept_scores[concept_id]["relevance"] += 2
                    
                    # Boost for concept type relevance
                    concept_type = concept.get('concept_type', '').lower()
                    if query_type != "general" and query_type in concept_type:
                        concept_scores[concept_id]["relevance"] += 1
        
        # Filter out fragmented or low-quality results
        filtered_concepts = {}
        for concept_id, data in concept_scores.items():
            concept = data["concept"]
            concept_name = concept.get('concept_name', '')
            
            # Skip very short or fragmented concept names
            if len(concept_name) < 15:
                continue
                
            # Skip if concept name starts with technical codes (A1:, Q3:, etc.)
            if concept_name.startswith(('A1:', 'A2:', 'A3:', 'Q1:', 'Q2:', 'Q3:', 'A:', 'Q:', '")', '1.', '2.', '3.')):
                continue
            
            # Skip fragmented sentences or incomplete thoughts
            if concept_name.count('.') > 3 or concept_name.count('?') > 1:
                continue
                
            # Skip if it looks like a question-answer fragment
            if 'according to' in concept_name.lower() or 'what is' in concept_name.lower():
                continue
                
            # Prefer concepts with meaningful descriptions
            descriptions = concept.get('descriptions', [])
            if descriptions:
                first_desc = descriptions[0] if descriptions else ""
                # Skip if description is just "Answer from Ayurvedic QA dataset"
                if "Answer from Ayurvedic QA dataset" in first_desc or "Question from Ayurvedic QA dataset" in first_desc:
                    continue
                    
            # Boost relevance for well-formed concepts
            if len(concept_name) > 30 and not any(char in concept_name for char in [':', '?', '(', ')']):
                data["relevance"] += 2
                
            filtered_concepts[concept_id] = data
        
        # If no results after filtering, try a less aggressive filter
        if not filtered_concepts and concept_scores:
            logger.info("No results after strict filtering, trying less aggressive filter...")
            for concept_id, data in concept_scores.items():
                concept = data["concept"]
                concept_name = concept.get('concept_name', '')
                
                # Less aggressive filtering - only skip obvious fragments
                if len(concept_name) < 8:
                    continue
                    
                # Skip only the most obvious QA fragments
                if concept_name.startswith(('A1:', 'A2:', 'A3:', 'Q1:', 'Q2:', 'Q3:')):
                    continue
                
                # Skip if description is just the generic QA dataset text
                descriptions = concept.get('descriptions', [])
                if descriptions:
                    first_desc = descriptions[0] if descriptions else ""
                    if first_desc == "Answer from Ayurvedic QA dataset" or first_desc == "Question from Ayurvedic QA dataset":
                        continue
                
                filtered_concepts[concept_id] = data
                
                # Limit to top results for less aggressive filtering
                if len(filtered_concepts) >= limit:
                    break
        
        # Sort by combined score and relevance
        sorted_concepts = sorted(
            filtered_concepts.values(),
            key=lambda x: (x["relevance"], x["score"]),
            reverse=True
        )
        
        # Format results
        results = []
        for item in sorted_concepts[:limit]:  # Limit results
            concept = item["concept"]
            formatted_concept = self._format_concept(concept, user_role)
            results.append(formatted_concept)
        
        return results
    
    def _handle_cross_domain_queries(self, query_text: str) -> List[Dict[str, Any]]:
        """Handle cross-domain mapping queries."""
        query_lower = query_text.lower()
        
        # Check if this is a cross-domain query
        cross_domain_patterns = [
            "ayurvedic equivalent of",
            "natural alternative to",
            "herbal substitute for",
            "ayurvedic medicine for",
            "allopathic equivalent of",
            "modern medicine for",
            "western medicine for"
        ]
        
        is_cross_domain_query = any(pattern in query_lower for pattern in cross_domain_patterns)
        
        if not is_cross_domain_query:
            return []
        
        try:
            # Determine query type
            if any(pattern in query_lower for pattern in ["ayurvedic equivalent", "natural alternative", "herbal substitute"]):
                query_type = "allopathic_to_ayurvedic"
            else:
                query_type = "ayurvedic_to_allopathic"
            
            # Query the cross-domain mapper
            mapping_query = MappingQuery(
                query_text=query_text,
                query_type=query_type,
                include_context=True
            )
            
            mapping_response = self.cross_domain_mapper.query_mappings(mapping_query)
            
            # Format mappings for response
            formatted_mappings = []
            for mapping in mapping_response.mappings:
                formatted_mapping = {
                    "biomedical_concept": mapping.allopathic_medicine,
                    "ayurvedic_concept": ", ".join(mapping.ayurvedic_herbs),
                    "allopathic_medicine": mapping.allopathic_medicine,  # Keep for backward compatibility
                    "ayurvedic_herbs": mapping.ayurvedic_herbs,  # Keep for backward compatibility
                    "disease_context": mapping.disease_context,
                    "formulation": mapping.formulation,
                    "confidence_score": mapping.confidence_score,
                    "source": mapping.source,
                    "source_evidence": [f"Dataset mapping for {mapping.disease_context}" if mapping.disease_context else "AyurGenixAI dataset mapping"],
                    "mapping_type": query_type
                }
                formatted_mappings.append(formatted_mapping)
            
            return formatted_mappings
            
        except Exception as e:
            logger.error(f"Failed to handle cross-domain query: {str(e)}")
            return []
    
    def _handle_fundamental_queries(self, query_lower: str, user_role: str) -> Optional[List[Dict[str, Any]]]:
        """Handle fundamental Ayurvedic concept queries with curated responses."""
        
        # Doshas query
        if any(word in query_lower for word in ["dosha", "doshas", "three dosha", "vata pitta kapha"]):
            return [{
                "concept_name": "The Three Doshas in Ayurveda",
                "concept_type": "fundamental_concept",
                "descriptions": [
                    "The three doshas are the fundamental energetic principles that govern all physiological and psychological processes in the body according to Ayurveda.",
                    "Vata (Air + Space): Controls movement, circulation, breathing, and nervous system functions. When balanced, promotes creativity and flexibility.",
                    "Pitta (Fire + Water): Governs metabolism, digestion, body temperature, and transformation. When balanced, promotes intelligence and courage.",
                    "Kapha (Earth + Water): Provides structure, stability, immunity, and lubrication. When balanced, promotes strength and compassion.",
                    "Each person has a unique constitution (Prakriti) with varying proportions of these three doshas, and health is maintained by keeping them in balance."
                ],
                "ayurvedic_terms": ["Vata", "Pitta", "Kapha", "Tridosha", "Prakriti"],
                "biomedical_terms": ["Constitutional types", "Metabolic principles"],
                "sources": ["Classical Ayurvedic texts", "Charaka Samhita", "Sushruta Samhita"],
                "properties": {
                    "vata_qualities": "Dry, light, cold, rough, subtle, mobile",
                    "pitta_qualities": "Hot, sharp, light, liquid, spreading, oily",
                    "kapha_qualities": "Heavy, slow, cool, oily, smooth, stable"
                } if user_role == "practitioner" else {}
            }]
        
        # Panchakarma query
        elif any(word in query_lower for word in ["panchakarma", "pancha karma", "five actions", "detox"]):
            return [{
                "concept_name": "Panchakarma - The Five Purification Actions",
                "concept_type": "treatment_system",
                "descriptions": [
                    "Panchakarma is Ayurveda's premier detoxification and rejuvenation program consisting of five therapeutic actions.",
                    "Vamana (Therapeutic vomiting): Eliminates excess Kapha from the respiratory and digestive systems.",
                    "Virechana (Purgation): Removes excess Pitta through controlled bowel elimination.",
                    "Basti (Medicated enemas): Balances Vata dosha and cleanses the colon.",
                    "Nasya (Nasal administration): Clears the head and neck region of accumulated toxins.",
                    "Raktamokshana (Bloodletting): Purifies blood and removes Pitta-related toxins."
                ],
                "ayurvedic_terms": ["Panchakarma", "Vamana", "Virechana", "Basti", "Nasya", "Raktamokshana"],
                "biomedical_terms": ["Detoxification", "Purification therapy"],
                "sources": ["Charaka Samhita", "Sushruta Samhita", "Ashtanga Hridaya"]
            }]
        
        # Ayurveda basics query
        elif any(word in query_lower for word in ["what is ayurveda", "ayurveda definition", "ayurvedic medicine"]):
            return [{
                "concept_name": "Ayurveda - The Science of Life",
                "concept_type": "medical_system",
                "descriptions": [
                    "Ayurveda is a 5,000-year-old system of natural healing that originated in India, meaning 'knowledge of life' in Sanskrit.",
                    "It focuses on preventing disease and promoting health through lifestyle practices, dietary guidelines, herbal remedies, and therapeutic treatments.",
                    "Based on the principle that health results from balance between mind, body, and consciousness, while disease arises from imbalance.",
                    "Emphasizes individualized treatment based on one's unique constitution (Prakriti) and current state of imbalance (Vikriti).",
                    "Integrates physical, mental, emotional, and spiritual aspects of health for comprehensive wellness."
                ],
                "ayurvedic_terms": ["Ayurveda", "Prakriti", "Vikriti", "Swasthya"],
                "biomedical_terms": ["Traditional medicine", "Holistic healthcare", "Preventive medicine"],
                "sources": ["Charaka Samhita", "Sushruta Samhita", "Ashtanga Hridaya", "Kashyapa Samhita"]
            }]
        
        # Turmeric query
        elif any(word in query_lower for word in ["turmeric", "haldi", "haridra", "curcuma"]):
            return [{
                "concept_name": "Turmeric (Curcuma longa) - The Golden Spice",
                "concept_type": "herb",
                "descriptions": [
                    "Turmeric is one of the most revered herbs in Ayurveda, known as 'Haridra' in Sanskrit, meaning 'the golden one'.",
                    "Contains curcumin, a powerful anti-inflammatory and antioxidant compound that gives turmeric its distinctive golden color.",
                    "Traditionally used for digestive health, wound healing, skin conditions, joint support, and liver detoxification.",
                    "Balances all three doshas but particularly beneficial for Kapha and Vata conditions.",
                    "Modern research confirms its anti-inflammatory, antimicrobial, and immune-supporting properties."
                ],
                "ayurvedic_terms": ["Haridra", "Haldi", "Rajani", "Yoshitapriya"],
                "biomedical_terms": ["Curcuma longa", "Curcumin", "Anti-inflammatory", "Antioxidant"],
                "sources": ["Charaka Samhita", "Sushruta Samhita", "Modern research"],
                "properties": {
                    "taste": "Bitter, pungent",
                    "potency": "Heating",
                    "post_digestive_effect": "Pungent",
                    "doshas_affected": "Reduces Kapha and Vata, may increase Pitta in excess"
                } if user_role == "practitioner" else {}
            }]
        
        # Ginger query
        elif any(word in query_lower for word in ["ginger", "adrak", "shunthi"]):
            return [{
                "concept_name": "Ginger (Zingiber officinale) - The Universal Medicine",
                "concept_type": "herb",
                "descriptions": [
                    "Ginger is called 'Vishwabheshaj' (universal medicine) in Ayurveda due to its wide-ranging therapeutic properties.",
                    "Excellent for digestive issues, nausea, respiratory conditions, and circulation enhancement.",
                    "Fresh ginger (Adrak) is more cooling and better for Pitta conditions, while dried ginger (Shunthi) is heating and better for Vata and Kapha.",
                    "Stimulates digestive fire (Agni), reduces toxins (Ama), and enhances absorption of other herbs.",
                    "Modern research confirms its anti-nausea, anti-inflammatory, and digestive benefits."
                ],
                "ayurvedic_terms": ["Adrak", "Shunthi", "Vishwabheshaj", "Ardraka"],
                "biomedical_terms": ["Zingiber officinale", "Gingerols", "Digestive stimulant"],
                "sources": ["Ayurvedic pharmacopoeia", "Traditional use", "Clinical studies"]
            }]
        
        # Hypertension/high blood pressure query
        elif any(phrase in query_lower for phrase in ["hypertension", "high blood pressure", "blood pressure", "raktagata vata"]):
            return [{
                "concept_name": "Ayurvedic Approach to Hypertension (Raktagata Vata)",
                "concept_type": "disease_management",
                "descriptions": [
                    "In Ayurveda, hypertension is understood as 'Raktagata Vata' - a condition where aggravated Vata dosha affects blood circulation.",
                    "The condition often involves Pitta dosha as well, creating heat and inflammation in the cardiovascular system.",
                    "Treatment focuses on calming Vata, cooling Pitta, and supporting healthy circulation through herbs and lifestyle modifications.",
                    "Key herbs include Arjuna for heart support, Brahmi for stress reduction, and Punarnava for fluid balance.",
                    "Stress management through meditation, pranayama, and yoga is essential for long-term blood pressure control.",
                    "Dietary recommendations include reducing salt, avoiding spicy foods, and incorporating cooling, calming foods."
                ],
                "ayurvedic_terms": ["Raktagata Vata", "Vyana Vata", "Sadhaka Pitta", "Hridaya"],
                "biomedical_terms": ["Hypertension", "Blood pressure", "Cardiovascular health"],
                "sources": ["Charaka Samhita", "Modern Ayurvedic cardiology", "Clinical studies"],
                "properties": {
                    "primary_dosha": "Vata with Pitta involvement",
                    "key_herbs": "Arjuna, Brahmi, Punarnava, Jatamansi",
                    "lifestyle_approach": "Stress reduction, regular exercise, meditation",
                    "dietary_guidelines": "Low salt, cooling foods, avoid spicy/hot foods"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.89
            }]

        # Arthritis/joint pain query
        elif any(phrase in query_lower for phrase in ["arthritis", "joint pain", "amavata", "sandhivata"]):
            return [{
                "concept_name": "Ayurvedic Approach to Arthritis (Amavata/Sandhivata)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Ayurveda recognizes two main types of arthritis: Amavata (rheumatoid arthritis) and Sandhivata (osteoarthritis).",
                    "Amavata involves Ama (toxins) combined with aggravated Vata, causing inflammation and pain in joints.",
                    "Sandhivata is primarily a Vata disorder causing degeneration of joint tissues and cartilage.",
                    "Treatment includes detoxification (Panchakarma), anti-inflammatory herbs, and Vata-pacifying therapies.",
                    "Key herbs include Guggulu for inflammation, Shallaki for joint support, and Rasna for pain relief.",
                    "External therapies like oil massage (Abhyanga) and medicated steam (Swedana) provide significant relief."
                ],
                "ayurvedic_terms": ["Amavata", "Sandhivata", "Sandhi", "Ama", "Vata dosha"],
                "biomedical_terms": ["Rheumatoid arthritis", "Osteoarthritis", "Joint inflammation"],
                "sources": ["Madhava Nidana", "Ayurvedic rheumatology", "Traditional treatments"],
                "properties": {
                    "primary_dosha": "Vata (with Ama in rheumatoid type)",
                    "key_herbs": "Guggulu, Shallaki, Rasna, Nirgundi",
                    "therapies": "Panchakarma, Abhyanga, Swedana, Basti",
                    "dietary_approach": "Anti-inflammatory diet, avoid Ama-forming foods"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.91
            }]

        # Asthma/respiratory issues query
        elif any(phrase in query_lower for phrase in ["asthma", "breathing problems", "tamaka shwasa", "shwasa roga"]):
            return [{
                "concept_name": "Ayurvedic Approach to Asthma (Tamaka Shwasa)",
                "concept_type": "disease_management",
                "descriptions": [
                    "In Ayurveda, asthma is called 'Tamaka Shwasa' and is primarily a Kapha-Vata disorder affecting the respiratory system.",
                    "Excess Kapha blocks the respiratory channels while aggravated Vata causes spasms and difficulty breathing.",
                    "Treatment focuses on reducing Kapha, calming Vata, and strengthening the respiratory system.",
                    "Key herbs include Vasaka for bronchodilation, Pushkarmool for respiratory strength, and Kantakari for Kapha reduction.",
                    "Pranayama (breathing exercises) and yoga are essential components of long-term asthma management.",
                    "Dietary modifications include avoiding cold, heavy, and mucus-forming foods while favoring warm, light meals."
                ],
                "ayurvedic_terms": ["Tamaka Shwasa", "Shwasa Roga", "Kapha-Vata", "Pranavahasrotas"],
                "biomedical_terms": ["Asthma", "Bronchospasm", "Respiratory inflammation"],
                "sources": ["Charaka Samhita", "Ayurvedic pulmonology", "Respiratory therapy texts"],
                "properties": {
                    "primary_dosha": "Kapha-Vata combination",
                    "key_herbs": "Vasaka, Pushkarmool, Kantakari, Bharangi",
                    "therapies": "Pranayama, steam inhalation, chest massage",
                    "dietary_approach": "Warm, light foods; avoid cold, heavy, dairy"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.88
            }]

        # Migraine/headache query
        elif any(phrase in query_lower for phrase in ["migraine", "headache", "shiroroga", "ardhavabhedaka"]):
            return [{
                "concept_name": "Ayurvedic Approach to Migraine (Ardhavabhedaka)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Migraine in Ayurveda is known as 'Ardhavabhedaka' (half-head pain) and is primarily a Vata-Pitta disorder.",
                    "Aggravated Vata causes neurological disturbances while increased Pitta creates heat and inflammation in the head region.",
                    "Treatment involves cooling Pitta, calming Vata, and addressing underlying digestive imbalances that often trigger migraines.",
                    "Key herbs include Brahmi for neurological support, Jatamansi for stress relief, and Shankhpushpi for mental clarity.",
                    "Lifestyle factors like regular sleep, stress management, and avoiding trigger foods are crucial for prevention.",
                    "Nasya (nasal therapy) and head massage with cooling oils provide immediate relief during acute episodes."
                ],
                "ayurvedic_terms": ["Ardhavabhedaka", "Shiroroga", "Vata-Pitta", "Nasya"],
                "biomedical_terms": ["Migraine", "Neurological headache", "Vascular headache"],
                "sources": ["Sushruta Samhita", "Ayurvedic neurology", "Headache management texts"],
                "properties": {
                    "primary_dosha": "Vata-Pitta combination",
                    "key_herbs": "Brahmi, Jatamansi, Shankhpushpi, Saraswatarishta",
                    "therapies": "Nasya, head massage, Shirodhara",
                    "lifestyle_factors": "Regular sleep, stress management, trigger avoidance"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.87
            }]

        # Gastritis/digestive issues query
        elif any(phrase in query_lower for phrase in ["gastritis", "stomach problems", "amlapitta", "urdhvaga amlapitta"]):
            return [{
                "concept_name": "Ayurvedic Approach to Gastritis (Amlapitta)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Gastritis in Ayurveda is called 'Amlapitta' and is primarily a Pitta disorder affecting the digestive system.",
                    "Excess Pitta creates hyperacidity, inflammation, and burning sensation in the stomach lining.",
                    "Treatment focuses on cooling and reducing Pitta while strengthening digestive fire (Agni) in a balanced way.",
                    "Key herbs include Amalaki for cooling acidity, Yashtimadhu for stomach lining protection, and Shatavari for healing.",
                    "Dietary management emphasizes cooling, alkaline foods while avoiding spicy, acidic, and fermented items.",
                    "Stress reduction and regular meal timing are essential for preventing gastric flare-ups."
                ],
                "ayurvedic_terms": ["Amlapitta", "Pitta dosha", "Agni", "Grahani"],
                "biomedical_terms": ["Gastritis", "Hyperacidity", "Peptic inflammation"],
                "sources": ["Charaka Samhita", "Ayurvedic gastroenterology", "Digestive health texts"],
                "properties": {
                    "primary_dosha": "Pitta with secondary Vata involvement",
                    "key_herbs": "Amalaki, Yashtimadhu, Shatavari, Kamadudha",
                    "dietary_approach": "Cooling foods, avoid spicy/acidic items",
                    "lifestyle_factors": "Regular meals, stress management, adequate rest"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.90
            }]

        # Insomnia/sleep disorders query
        elif any(phrase in query_lower for phrase in ["insomnia", "sleep problems", "anidra", "nidranasha"]):
            return [{
                "concept_name": "Ayurvedic Approach to Insomnia (Anidra)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Insomnia in Ayurveda is called 'Anidra' and is primarily caused by aggravated Vata dosha affecting the nervous system.",
                    "Excess Vata creates mental restlessness, anxiety, and inability to achieve deep, restorative sleep.",
                    "Treatment focuses on calming Vata, nourishing the nervous system, and establishing healthy sleep rhythms.",
                    "Key herbs include Ashwagandha for stress adaptation, Brahmi for mental calm, and Jatamansi for deep sleep.",
                    "Evening routines with oil massage, warm milk with spices, and meditation help prepare the body for rest.",
                    "Lifestyle modifications include regular sleep schedule, avoiding stimulants, and creating a peaceful sleep environment."
                ],
                "ayurvedic_terms": ["Anidra", "Nidranasha", "Vata dosha", "Majja dhatu"],
                "biomedical_terms": ["Insomnia", "Sleep disorder", "Sleep deprivation"],
                "sources": ["Charaka Samhita", "Ayurvedic psychiatry", "Sleep medicine texts"],
                "properties": {
                    "primary_dosha": "Vata with possible Pitta involvement",
                    "key_herbs": "Ashwagandha, Brahmi, Jatamansi, Shankhpushpi",
                    "therapies": "Abhyanga, Shirodhara, meditation",
                    "lifestyle_approach": "Regular sleep schedule, calming evening routine"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.86
            }]

        # Anxiety/mental health query
        elif any(phrase in query_lower for phrase in ["anxiety", "stress", "chittodvega", "mental stress"]):
            return [{
                "concept_name": "Ayurvedic Approach to Anxiety (Chittodvega)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Anxiety in Ayurveda is understood as 'Chittodvega' - disturbance of consciousness due to aggravated Vata and Rajas.",
                    "Excess Vata creates mental instability, restlessness, and fear, while increased Rajas causes agitation and worry.",
                    "Treatment focuses on calming Vata, reducing Rajas, and strengthening Ojas (vital essence) for mental stability.",
                    "Key herbs include Ashwagandha for stress adaptation, Brahmi for mental clarity, and Shankhpushpi for emotional balance.",
                    "Meditation, pranayama, and yoga are essential practices for long-term anxiety management and mental peace.",
                    "Lifestyle modifications include regular routine, adequate rest, and avoiding overstimulation from media and activities."
                ],
                "ayurvedic_terms": ["Chittodvega", "Vata dosha", "Rajas", "Ojas", "Satvavajaya"],
                "biomedical_terms": ["Anxiety disorder", "Stress response", "Mental health"],
                "sources": ["Charaka Samhita", "Ayurvedic psychiatry", "Mental health texts"],
                "properties": {
                    "primary_dosha": "Vata with Rajas involvement",
                    "key_herbs": "Ashwagandha, Brahmi, Shankhpushpi, Saraswatarishta",
                    "therapies": "Meditation, pranayama, Shirodhara",
                    "lifestyle_approach": "Regular routine, stress reduction, mindfulness"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.85
            }]

        # Depression/mood disorders query
        elif any(phrase in query_lower for phrase in ["depression", "mood disorder", "vishada", "manoavasada"]):
            return [{
                "concept_name": "Ayurvedic Approach to Depression (Vishada)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Depression in Ayurveda is called 'Vishada' and involves imbalance of all three doshas with predominant Kapha and Tamas.",
                    "Excess Kapha creates heaviness and lethargy, while increased Tamas causes mental darkness and lack of motivation.",
                    "Treatment focuses on reducing Kapha and Tamas while increasing Sattva (mental clarity) and Rajas (activity).",
                    "Key herbs include Brahmi for mental clarity, Mandukaparni for mood elevation, and Saraswatarishta for cognitive support.",
                    "Lifestyle interventions include regular exercise, sunlight exposure, social engagement, and purposeful activities.",
                    "Panchakarma therapies, especially Nasya and Shirodhara, help balance brain chemistry and improve mental state."
                ],
                "ayurvedic_terms": ["Vishada", "Manoavasada", "Kapha-Tamas", "Sattva", "Satvavajaya"],
                "biomedical_terms": ["Depression", "Mood disorder", "Mental health"],
                "sources": ["Charaka Samhita", "Ayurvedic psychiatry", "Mental wellness texts"],
                "properties": {
                    "primary_dosha": "Kapha with Tamas predominance",
                    "key_herbs": "Brahmi, Mandukaparni, Saraswatarishta, Medhya Rasayana",
                    "therapies": "Panchakarma, Nasya, Shirodhara, counseling",
                    "lifestyle_approach": "Regular exercise, sunlight, social engagement"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.84
            }]

        # Obesity/weight management query
        elif any(phrase in query_lower for phrase in ["obesity", "weight gain", "sthaulya", "medoroga"]):
            return [{
                "concept_name": "Ayurvedic Approach to Obesity (Sthaulya)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Obesity in Ayurveda is called 'Sthaulya' and is primarily a Kapha disorder with involvement of Meda dhatu (fat tissue).",
                    "Excess Kapha and weakened Agni (digestive fire) lead to accumulation of Ama (toxins) and excessive fat tissue formation.",
                    "Treatment focuses on strengthening Agni, reducing Kapha, and promoting healthy metabolism through herbs and lifestyle changes.",
                    "Key herbs include Guggulu for fat metabolism, Triphala for detoxification, and Vrikshamla for appetite control.",
                    "Dietary approach emphasizes light, warm foods with bitter and pungent tastes while avoiding heavy, sweet, and oily items.",
                    "Regular exercise, especially yoga and walking, combined with intermittent fasting helps restore healthy weight."
                ],
                "ayurvedic_terms": ["Sthaulya", "Medoroga", "Kapha dosha", "Meda dhatu", "Agni mandya"],
                "biomedical_terms": ["Obesity", "Metabolic disorder", "Weight management"],
                "sources": ["Charaka Samhita", "Ayurvedic metabolism texts", "Weight management studies"],
                "properties": {
                    "primary_dosha": "Kapha with Agni mandya",
                    "key_herbs": "Guggulu, Triphala, Vrikshamla, Punarnava",
                    "dietary_approach": "Light, warm foods; bitter/pungent tastes",
                    "lifestyle_factors": "Regular exercise, yoga, intermittent fasting"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.88
            }]

        # Anemia/blood disorders query
        elif any(phrase in query_lower for phrase in ["anemia", "low hemoglobin", "pandu roga", "raktalpata"]):
            return [{
                "concept_name": "Ayurvedic Approach to Anemia (Pandu Roga)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Anemia in Ayurveda is called 'Pandu Roga' and involves vitiation of Pitta dosha affecting Rakta dhatu (blood tissue).",
                    "Weakened Agni leads to poor nutrition absorption and inadequate formation of healthy blood tissue.",
                    "Treatment focuses on strengthening Agni, nourishing Rakta dhatu, and correcting underlying digestive imbalances.",
                    "Key herbs include Punarnava for blood formation, Amalaki for iron absorption, and Mandur Bhasma for hemoglobin increase.",
                    "Dietary recommendations include iron-rich foods, vitamin C sources, and avoiding tea/coffee with meals.",
                    "Lifestyle modifications include adequate rest, gentle exercise, and stress management to support blood formation."
                ],
                "ayurvedic_terms": ["Pandu Roga", "Rakta dhatu", "Pitta dosha", "Agni mandya"],
                "biomedical_terms": ["Anemia", "Iron deficiency", "Hemoglobin deficiency"],
                "sources": ["Charaka Samhita", "Ayurvedic hematology", "Blood disorder texts"],
                "properties": {
                    "primary_dosha": "Pitta with Agni involvement",
                    "key_herbs": "Punarnava, Amalaki, Mandur Bhasma, Lohasava",
                    "dietary_approach": "Iron-rich foods, vitamin C, avoid tea with meals",
                    "lifestyle_factors": "Adequate rest, gentle exercise, stress management"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.87
            }]

        # Bronchitis/respiratory infection query
        elif any(phrase in query_lower for phrase in ["bronchitis", "chest congestion", "kasa", "kaphajavyadhi"]):
            return [{
                "concept_name": "Ayurvedic Approach to Bronchitis (Kasa)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Bronchitis in Ayurveda is understood as 'Kasa' and is primarily a Kapha disorder affecting the respiratory system.",
                    "Excess Kapha accumulates in the lungs and bronchi, causing congestion, cough, and difficulty in breathing.",
                    "Treatment focuses on reducing Kapha, clearing respiratory channels, and strengthening lung function.",
                    "Key herbs include Vasaka for expectoration, Kantakari for bronchodilation, and Pushkarmool for respiratory strength.",
                    "Steam inhalation with eucalyptus or tulsi helps clear congestion and reduce inflammation.",
                    "Dietary modifications include warm, light foods and avoiding cold, heavy, dairy products that increase Kapha."
                ],
                "ayurvedic_terms": ["Kasa", "Kapha dosha", "Pranavahasrotas", "Kaphajavyadhi"],
                "biomedical_terms": ["Bronchitis", "Respiratory infection", "Bronchial inflammation"],
                "sources": ["Charaka Samhita", "Ayurvedic pulmonology", "Respiratory therapy"],
                "properties": {
                    "primary_dosha": "Kapha with possible Vata involvement",
                    "key_herbs": "Vasaka, Kantakari, Pushkarmool, Bharangi",
                    "therapies": "Steam inhalation, chest massage, pranayama",
                    "dietary_approach": "Warm, light foods; avoid cold, dairy"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.89
            }]

        # Constipation/digestive issues query
        elif any(phrase in query_lower for phrase in ["constipation", "bowel problems", "vibandha", "malabaddhata"]):
            return [{
                "concept_name": "Ayurvedic Approach to Constipation (Vibandha)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Constipation in Ayurveda is called 'Vibandha' and is primarily a Vata disorder affecting the large intestine.",
                    "Aggravated Vata causes dryness and irregular movement in the colon, leading to hard, difficult-to-pass stools.",
                    "Treatment focuses on pacifying Vata, adding moisture and lubrication, and restoring normal bowel function.",
                    "Key herbs include Triphala for gentle laxative action, Isabgol for bulk and moisture, and Castor oil for lubrication.",
                    "Dietary recommendations include fiber-rich foods, adequate water intake, and healthy fats like ghee.",
                    "Lifestyle modifications include regular meal times, adequate exercise, and establishing consistent bowel habits."
                ],
                "ayurvedic_terms": ["Vibandha", "Malabaddhata", "Vata dosha", "Purishavahasrotas"],
                "biomedical_terms": ["Constipation", "Bowel dysfunction", "Intestinal motility"],
                "sources": ["Charaka Samhita", "Ayurvedic gastroenterology", "Digestive health"],
                "properties": {
                    "primary_dosha": "Vata with possible Kapha involvement",
                    "key_herbs": "Triphala, Isabgol, Castor oil, Haritaki",
                    "dietary_approach": "High fiber, adequate water, healthy fats",
                    "lifestyle_factors": "Regular meals, exercise, consistent routine"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.91
            }]

        # Diarrhea/loose motions query
        elif any(phrase in query_lower for phrase in ["diarrhea", "loose motions", "atisara", "pravahika"]):
            return [{
                "concept_name": "Ayurvedic Approach to Diarrhea (Atisara)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Diarrhea in Ayurveda is called 'Atisara' and can involve any of the three doshas, with Pitta being most common.",
                    "Pitta-type diarrhea involves heat and inflammation, Vata-type involves irregular motions, and Kapha-type involves mucus.",
                    "Treatment varies based on the dosha involved but generally focuses on digestive restoration and fluid balance.",
                    "Key herbs include Kutaja for antimicrobial action, Bilva for astringent properties, and Musta for digestive balance.",
                    "Dietary management includes easily digestible foods, adequate fluid replacement, and avoiding irritating substances.",
                    "Rest and gradual return to normal diet helps restore digestive function and prevent complications."
                ],
                "ayurvedic_terms": ["Atisara", "Pravahika", "Tridosha", "Grahani dosha"],
                "biomedical_terms": ["Diarrhea", "Gastroenteritis", "Intestinal inflammation"],
                "sources": ["Charaka Samhita", "Ayurvedic gastroenterology", "Digestive disorders"],
                "properties": {
                    "primary_dosha": "Variable (Pitta, Vata, or Kapha)",
                    "key_herbs": "Kutaja, Bilva, Musta, Dadimashtak",
                    "dietary_approach": "Light, easily digestible foods, fluid replacement",
                    "management": "Rest, gradual diet progression, hydration"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.88
            }]

        # Eczema/skin disorders query
        elif any(phrase in query_lower for phrase in ["eczema", "skin problems", "vicharchika", "kushtha"]):
            return [{
                "concept_name": "Ayurvedic Approach to Eczema (Vicharchika)",
                "concept_type": "disease_management",
                "descriptions": [
                    "Eczema in Ayurveda is called 'Vicharchika' and is primarily a Pitta-Kapha disorder affecting the skin.",
                    "Vitiated Pitta creates heat and inflammation while aggravated Kapha causes itching, oozing, and thickening of skin.",
                    "Treatment focuses on cooling Pitta, reducing Kapha, and purifying blood through detoxification and herbs.",
                    "Key herbs include Neem for antimicrobial action, Manjistha for blood purification, and Turmeric for anti-inflammatory effects.",
                    "External applications of cooling oils and herbal pastes provide symptomatic relief and healing.",
                    "Dietary modifications include avoiding spicy, sour, and fermented foods while favoring cooling, bitter tastes."
                ],
                "ayurvedic_terms": ["Vicharchika", "Kushtha", "Pitta-Kapha", "Raktashodhana"],
                "biomedical_terms": ["Eczema", "Atopic dermatitis", "Skin inflammation"],
                "sources": ["Charaka Samhita", "Ayurvedic dermatology", "Skin disorder texts"],
                "properties": {
                    "primary_dosha": "Pitta-Kapha combination",
                    "key_herbs": "Neem, Manjistha, Turmeric, Khadira",
                    "external_therapy": "Cooling oils, herbal pastes, medicated baths",
                    "dietary_approach": "Cooling foods, avoid spicy/sour/fermented"
                } if user_role == "practitioner" else {},
                "confidence_score": 0.86
            }]

        # Diabetes/blood sugar query
        elif any(phrase in query_lower for phrase in ["diabetes", "blood sugar", "diabetes treatment", "ayurvedic treatment for diabetes", "madhumeha"]):
            return [{
                "concept_name": "Ayurvedic Approach to Diabetes (Madhumeha)",
                "concept_type": "disease_management",
                "descriptions": [
                    "In Ayurveda, diabetes is known as 'Madhumeha' (sweet urine) and is primarily considered a Kapha disorder with Vata and Pitta involvement.",
                    "The condition arises from impaired metabolism (Agni mandya) leading to accumulation of sweet toxins (Ama) in the body.",
                    "Treatment focuses on strengthening digestive fire, reducing Kapha, and supporting pancreatic function through herbs and lifestyle modifications.",
                    "Key herbs include Gudmar (Gymnema sylvestre) for blood sugar control, Jamun seeds for pancreatic support, and Bitter melon for natural insulin-like effects.",
                    "Dietary management emphasizes bitter and astringent tastes while avoiding sweet, heavy, and oily foods that aggravate Kapha.",
                    "Regular exercise, stress management, and maintaining proper sleep cycles are essential components of holistic diabetes management."
                ],
                "ayurvedic_terms": ["Madhumeha", "Prameha", "Kapha dosha", "Agni mandya", "Ama"],
                "biomedical_terms": ["Type 2 diabetes", "Blood glucose", "Insulin resistance", "Metabolic disorder"],
                "sources": ["Charaka Samhita", "Sushruta Samhita", "Modern Ayurvedic research"],
                "properties": {
                    "primary_dosha": "Kapha with Vata-Pitta involvement",
                    "key_herbs": "Gudmar, Jamun, Karela, Methi, Haridra",
                    "dietary_approach": "Bitter and astringent tastes, avoid sweet/heavy foods",
                    "lifestyle_factors": "Regular exercise, stress management, proper sleep"
                } if user_role == "practitioner" else {},
                "relationships": [
                    {"type": "treated_with", "target_concept": "Gudmar (Gymnema sylvestre)"},
                    {"type": "managed_by", "target_concept": "Kapha-reducing diet"},
                    {"type": "supported_by", "target_concept": "Regular exercise"}
                ],
                "confidence_score": 0.92
            }]

        # Digestive health query
        elif any(phrase in query_lower for phrase in ["digestive health", "digestion", "agni", "digestive fire"]):
            return [{
                "concept_name": "Ayurvedic Approach to Digestive Health",
                "concept_type": "health_approach",
                "descriptions": [
                    "In Ayurveda, digestive health is considered the foundation of overall wellness, centered around the concept of Agni (digestive fire).",
                    "Strong Agni ensures proper digestion, absorption, and elimination while preventing the formation of Ama (toxins).",
                    "Key principles include eating according to your constitution, maintaining regular meal times, and consuming warm, freshly prepared foods.",
                    "Digestive spices like ginger, cumin, coriander, and fennel are used to kindle Agni and improve digestion.",
                    "Lifestyle factors such as mindful eating, proper food combining, and avoiding overeating are equally important."
                ],
                "ayurvedic_terms": ["Agni", "Ama", "Pachana", "Ahara", "Vihar"],
                "biomedical_terms": ["Digestive enzymes", "Gut health", "Metabolism"],
                "sources": ["Charaka Samhita", "Traditional Ayurvedic practice"]
            }]
        
        # Stress and anxiety query
        elif any(phrase in query_lower for phrase in ["stress", "anxiety", "mental health", "mind balance"]):
            return [{
                "concept_name": "Ayurvedic Approach to Stress and Mental Health",
                "concept_type": "health_approach", 
                "descriptions": [
                    "Ayurveda views stress and anxiety as primarily Vata imbalances affecting the nervous system and mind.",
                    "Treatment focuses on grounding practices, nourishing foods, and herbs that calm the nervous system.",
                    "Key herbs include Ashwagandha for stress adaptation, Brahmi for mental clarity, and Jatamansi for anxiety.",
                    "Lifestyle recommendations include regular routines, adequate sleep, meditation, and gentle yoga practices.",
                    "Pranayama (breathing exercises) and meditation are considered essential for mental balance and stress reduction."
                ],
                "ayurvedic_terms": ["Satvavajaya Chikitsa", "Medhya Rasayana", "Pranayama"],
                "biomedical_terms": ["Adaptogenic herbs", "Stress management", "Anxiety relief"],
                "sources": ["Ayurvedic psychology", "Traditional practice"]
            }]
        
        # Immunity query
        elif any(phrase in query_lower for phrase in ["immunity", "immune system", "ojas", "resistance"]):
            return [{
                "concept_name": "Ayurvedic Approach to Immunity and Ojas",
                "concept_type": "health_approach",
                "descriptions": [
                    "Ayurveda describes immunity through the concept of Ojas, the subtle essence that provides strength, vitality, and disease resistance.",
                    "Strong Ojas results from proper digestion, quality sleep, balanced emotions, and spiritual practices.",
                    "Rasayana (rejuvenative) herbs like Chyawanprash, Amalaki, and Guduchi are used to build Ojas and enhance immunity.",
                    "Lifestyle factors include regular exercise, stress management, adequate rest, and maintaining emotional balance.",
                    "Seasonal routines (Ritucharya) help maintain immunity by adapting to environmental changes."
                ],
                "ayurvedic_terms": ["Ojas", "Rasayana", "Vyadhikshamatva", "Ritucharya"],
                "biomedical_terms": ["Immune system", "Antioxidants", "Adaptogenic herbs"],
                "sources": ["Classical Ayurvedic texts", "Rasayana therapy"]
            }]
        
        # Ashwagandha query
        elif any(word in query_lower for word in ["ashwagandha", "withania", "winter cherry"]):
            return [{
                "concept_name": "Ashwagandha (Withania somnifera) - The Strength of a Horse",
                "concept_type": "herb",
                "descriptions": [
                    "Ashwagandha, meaning 'smell of horse', is named for its distinctive odor and its ability to impart the strength and vitality of a horse.",
                    "Premier adaptogenic herb in Ayurveda, helping the body manage stress and maintain energy balance.",
                    "Classified as a Rasayana (rejuvenative) herb, promoting longevity, vitality, and mental clarity.",
                    "Particularly beneficial for nervous system support, sleep quality, immune function, and physical strength.",
                    "Modern research shows significant benefits for stress reduction, cortisol regulation, and cognitive function."
                ],
                "ayurvedic_terms": ["Ashwagandha", "Varaha Karni", "Hayagandha"],
                "biomedical_terms": ["Withania somnifera", "Adaptogen", "Withanolides"],
                "sources": ["Classical Ayurvedic texts", "Modern clinical trials"]
            }]
        
        # Neem query
        elif any(word in query_lower for word in ["neem", "margosa", "nimba"]):
            return [{
                "concept_name": "Neem (Azadirachta indica) - Nature's Pharmacy",
                "concept_type": "herb",
                "descriptions": [
                    "Neem is revered in Ayurveda as 'Sarva Roga Nivarini' (the curer of all ailments) due to its wide-ranging therapeutic properties.",
                    "Powerful antimicrobial, antifungal, and anti-inflammatory herb used for skin conditions, dental health, and immune support.",
                    "Traditionally used for blood purification, diabetes management, and as a natural pesticide and preservative.",
                    "All parts of the neem tree are medicinal - leaves, bark, seeds, and oil each have specific therapeutic applications.",
                    "Modern research confirms its antibacterial, antiviral, and immune-modulating properties."
                ],
                "ayurvedic_terms": ["Nimba", "Sarva Roga Nivarini", "Arishta"],
                "biomedical_terms": ["Azadirachta indica", "Antimicrobial", "Azadirachtin"],
                "sources": ["Traditional Ayurvedic use", "Modern research"]
            }]
        
        # Brahmi query
        elif any(word in query_lower for word in ["brahmi", "bacopa", "memory", "brain health"]):
            return [{
                "concept_name": "Brahmi (Bacopa monnieri) - The Brain Tonic",
                "concept_type": "herb",
                "descriptions": [
                    "Brahmi is considered the premier brain tonic in Ayurveda, named after Brahma, the creator god, symbolizing divine consciousness.",
                    "Classified as a Medhya Rasayana (brain rejuvenative), specifically enhancing memory, learning, and cognitive function.",
                    "Traditionally used for anxiety, depression, epilepsy, and to promote mental clarity and spiritual awareness.",
                    "Balances both Vata and Pitta doshas, making it suitable for various nervous system imbalances.",
                    "Modern research shows significant benefits for memory enhancement, stress reduction, and neuroprotection."
                ],
                "ayurvedic_terms": ["Brahmi", "Medhya Rasayana", "Saraswati"],
                "biomedical_terms": ["Bacopa monnieri", "Nootropic", "Bacosides"],
                "sources": ["Classical Ayurvedic texts", "Cognitive research"]
            }]
        
        return None
    
    def _format_concept(self, concept: Dict[str, Any], user_role: str) -> Dict[str, Any]:
        """Format concept based on user role."""
        formatted = {
            "concept_name": concept.get('concept_name', concept.get('name', 'Unknown')),
            "concept_type": concept.get('concept_type', concept.get('type', 'general')),
            "descriptions": concept.get('descriptions', [])[:3],  # Limit descriptions
            "ayurvedic_terms": concept.get('ayurvedic_terms', []),
            "biomedical_terms": concept.get('biomedical_terms', []),
            "sources": concept.get('sources', [])
        }
        
        # Add role-specific information
        if user_role == "practitioner":
            formatted.update({
                "properties": concept.get('properties', {}),
                "contraindications": concept.get('contraindications', []),
                "dosage_info": concept.get('dosage_info', {}),
                "interactions": concept.get('interactions', [])
            })
        
        return formatted
    
    def _generate_warnings(self, user_role: str, concepts: List[Dict[str, Any]]) -> List[str]:
        """Generate appropriate warnings based on user role."""
        warnings = []
        
        if user_role == "general":
            warnings.extend([
                "This information is for educational purposes only.",
                "Always consult with qualified healthcare professionals before using any herbs or treatments.",
                "Do not use this information to self-diagnose or self-treat medical conditions."
            ])
        else:
            warnings.extend([
                "This information is intended for qualified healthcare practitioners.",
                "Always consider individual patient factors and contraindications.",
                "Verify dosages and interactions with current medical literature."
            ])
        
        return warnings
    
    def get_concept_by_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific concept by ID."""
        self._ensure_loaded()
        return self._concept_index.get(concept_id)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        self._ensure_loaded()
        
        # Handle different data formats
        concepts_data = self._knowledge_data.get('concepts', {})
        
        if isinstance(concepts_data, dict):
            concepts = list(concepts_data.values())
        else:
            concepts = concepts_data
        
        concept_types = {}
        
        for concept in concepts:
            concept_type = concept.get('concept_type', concept.get('type', 'unknown'))
            concept_types[concept_type] = concept_types.get(concept_type, 0) + 1
        
        return {
            "total_concepts": len(concepts),
            "concept_types": concept_types,
            "source_distribution": {"compiled": len(concepts)},
            "has_cross_domain_mapper": False,
            "max_results": 10,
            "indexed_terms": len(self._term_index) if self._term_index else 0
        }
    
    def refresh_knowledge_base(self) -> bool:
        """Refresh the knowledge base."""
        try:
            self._loaded = False
            self._knowledge_data = None
            self._concept_index = None
            self._term_index = None
            self._ensure_loaded()
            return True
        except Exception as e:
            logger.error(f"Failed to refresh knowledge base: {str(e)}")
            return False

# Global instance for reuse
_global_knowledge_system = None

def get_knowledge_system() -> OptimizedIntegratedKnowledgeSystem:
    """Get or create global knowledge system instance."""
    global _global_knowledge_system
    if _global_knowledge_system is None:
        _global_knowledge_system = OptimizedIntegratedKnowledgeSystem()
    return _global_knowledge_system