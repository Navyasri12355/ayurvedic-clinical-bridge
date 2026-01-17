"""
Optimized Safety Analyzer for herb-drug interaction detection.

This service provides efficient safety analysis using a comprehensive
database of known interactions and contraindications.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class InteractionSeverity(Enum):
    """Severity levels for herb-drug interactions."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

class UncertaintyLevel(Enum):
    """Uncertainty levels for safety assessments."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

@dataclass
class Interaction:
    """Herb-drug interaction information."""
    herb: str
    drug: str
    severity: InteractionSeverity
    description: str
    mechanism: str
    evidence_level: str
    confidence: float
    recommendations: List[str]

@dataclass
class SafetyAssessment:
    """Complete safety assessment result."""
    analysis_id: str
    interactions: List[Interaction]
    overall_risk: str
    confidence_score: float
    processing_time: float
    recommendations: List[str]
    uncertainty_assessment: Dict[str, Any]
    metadata: Dict[str, Any]

class OptimizedSafetyAnalyzer:
    """Optimized safety analyzer with comprehensive interaction database."""
    
    def __init__(self):
        """Initialize the safety analyzer."""
        self._load_interaction_database()
        logger.info("Initialized OptimizedSafetyAnalyzer")
    
    def _load_interaction_database(self):
        """Load comprehensive herb-drug interaction database."""
        # Comprehensive interaction database
        self.interactions_db = {
            # Turmeric interactions
            ("turmeric", "warfarin"): {
                "severity": InteractionSeverity.MODERATE,
                "description": "Turmeric may increase bleeding risk when combined with warfarin due to its anticoagulant properties.",
                "mechanism": "Curcumin in turmeric may enhance anticoagulant effects",
                "evidence_level": "moderate",
                "confidence": 0.78,
                "recommendations": [
                    "Monitor INR levels more frequently",
                    "Consider reducing turmeric dosage",
                    "Watch for signs of bleeding"
                ]
            },
            ("turmeric", "aspirin"): {
                "severity": InteractionSeverity.MODERATE,
                "description": "Combined use may increase bleeding risk due to additive antiplatelet effects.",
                "mechanism": "Both substances affect platelet aggregation",
                "evidence_level": "moderate",
                "confidence": 0.72,
                "recommendations": [
                    "Monitor for bleeding symptoms",
                    "Use lowest effective doses",
                    "Consider timing separation"
                ]
            },
            ("turmeric", "diabetes_medications"): {
                "severity": InteractionSeverity.LOW,
                "description": "Turmeric may enhance blood sugar lowering effects of diabetes medications.",
                "mechanism": "Curcumin may improve insulin sensitivity",
                "evidence_level": "low",
                "confidence": 0.65,
                "recommendations": [
                    "Monitor blood glucose levels",
                    "Adjust medication dosing if needed"
                ]
            },
            
            # Ginger interactions
            ("ginger", "warfarin"): {
                "severity": InteractionSeverity.MODERATE,
                "description": "Ginger may increase bleeding risk when combined with anticoagulants.",
                "mechanism": "Gingerols may inhibit platelet aggregation",
                "evidence_level": "moderate",
                "confidence": 0.75,
                "recommendations": [
                    "Monitor coagulation parameters",
                    "Limit ginger intake to culinary amounts",
                    "Watch for bleeding signs"
                ]
            },
            ("ginger", "diabetes_medications"): {
                "severity": InteractionSeverity.LOW,
                "description": "Ginger may enhance hypoglycemic effects of diabetes medications.",
                "mechanism": "May improve insulin sensitivity and glucose uptake",
                "evidence_level": "low",
                "confidence": 0.68,
                "recommendations": [
                    "Monitor blood sugar levels",
                    "Be aware of hypoglycemia symptoms"
                ]
            },
            
            # Ashwagandha interactions
            ("ashwagandha", "sedatives"): {
                "severity": InteractionSeverity.MODERATE,
                "description": "Ashwagandha may enhance sedative effects of CNS depressants.",
                "mechanism": "GABA-ergic activity may be enhanced",
                "evidence_level": "moderate",
                "confidence": 0.70,
                "recommendations": [
                    "Avoid driving or operating machinery",
                    "Start with lower doses",
                    "Monitor for excessive sedation"
                ]
            },
            ("ashwagandha", "immunosuppressants"): {
                "severity": InteractionSeverity.MODERATE,
                "description": "Ashwagandha may counteract immunosuppressive medications.",
                "mechanism": "Immune-stimulating properties may oppose immunosuppression",
                "evidence_level": "theoretical",
                "confidence": 0.60,
                "recommendations": [
                    "Consult with healthcare provider",
                    "Monitor immune function markers",
                    "Consider alternative adaptogens"
                ]
            },
            
            # Ginkgo interactions
            ("ginkgo", "anticoagulants"): {
                "severity": InteractionSeverity.HIGH,
                "description": "Ginkgo significantly increases bleeding risk with anticoagulants.",
                "mechanism": "Inhibits platelet-activating factor",
                "evidence_level": "high",
                "confidence": 0.85,
                "recommendations": [
                    "Avoid concurrent use",
                    "If used, monitor closely for bleeding",
                    "Consider alternative herbs"
                ]
            },
            
            # St. John's Wort interactions
            ("st_johns_wort", "antidepressants"): {
                "severity": InteractionSeverity.SEVERE,
                "description": "Risk of serotonin syndrome when combined with SSRIs or other antidepressants.",
                "mechanism": "Dual serotonin reuptake inhibition",
                "evidence_level": "high",
                "confidence": 0.90,
                "recommendations": [
                    "Avoid concurrent use",
                    "Taper one agent before starting the other",
                    "Monitor for serotonin syndrome symptoms"
                ]
            },
            ("st_johns_wort", "birth_control"): {
                "severity": InteractionSeverity.HIGH,
                "description": "St. John's Wort may reduce effectiveness of hormonal contraceptives.",
                "mechanism": "Induces CYP3A4 enzyme, increasing hormone metabolism",
                "evidence_level": "high",
                "confidence": 0.88,
                "recommendations": [
                    "Use additional contraceptive methods",
                    "Consider alternative herbs",
                    "Consult healthcare provider"
                ]
            },
            
            # Garlic interactions
            ("garlic", "anticoagulants"): {
                "severity": InteractionSeverity.MODERATE,
                "description": "Garlic supplements may increase bleeding risk with anticoagulants.",
                "mechanism": "Antiplatelet effects of allicin compounds",
                "evidence_level": "moderate",
                "confidence": 0.73,
                "recommendations": [
                    "Limit to culinary amounts",
                    "Monitor coagulation parameters",
                    "Discontinue before surgery"
                ]
            }
        }
        
        # Herb normalization mapping
        self.herb_aliases = {
            "turmeric": ["turmeric", "curcuma", "haldi", "haridra"],
            "ginger": ["ginger", "zingiber", "adrak", "shunthi"],
            "ashwagandha": ["ashwagandha", "withania", "winter cherry"],
            "ginkgo": ["ginkgo", "ginkgo biloba", "maidenhair tree"],
            "st_johns_wort": ["st john's wort", "st johns wort", "hypericum"],
            "garlic": ["garlic", "allium", "lasuna"]
        }
        
        # Drug normalization mapping
        self.drug_aliases = {
            "warfarin": ["warfarin", "coumadin", "jantoven"],
            "aspirin": ["aspirin", "acetylsalicylic acid", "asa"],
            "diabetes_medications": ["metformin", "insulin", "glipizide", "glyburide", "pioglitazone"],
            "sedatives": ["lorazepam", "diazepam", "alprazolam", "zolpidem", "temazepam"],
            "immunosuppressants": ["cyclosporine", "tacrolimus", "methotrexate", "prednisone"],
            "anticoagulants": ["warfarin", "heparin", "enoxaparin", "rivaroxaban", "apixaban"],
            "antidepressants": ["sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram"],
            "birth_control": ["ethinyl estradiol", "levonorgestrel", "norethindrone", "drospirenone"]
        }
    
    def analyze_safety(
        self,
        herbs: List[str],
        drugs: List[str],
        patient_factors: Optional[Dict[str, Any]] = None,
        include_uncertainty_assessment: bool = True,
        include_consultation_recommendations: bool = True
    ) -> SafetyAssessment:
        """Analyze safety of herb-drug combinations."""
        start_time = time.time()
        analysis_id = f"safety_{int(time.time() * 1000)}"
        
        # Normalize herb and drug names
        normalized_herbs = self._normalize_herbs(herbs)
        normalized_drugs = self._normalize_drugs(drugs)
        
        # Find interactions
        interactions = self._find_interactions(normalized_herbs, normalized_drugs)
        
        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(interactions)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(interactions, len(herbs), len(drugs))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(interactions, patient_factors)
        
        # Uncertainty assessment
        uncertainty_assessment = {}
        if include_uncertainty_assessment:
            uncertainty_assessment = self._assess_uncertainty(interactions, herbs, drugs)
        
        processing_time = time.time() - start_time
        
        return SafetyAssessment(
            analysis_id=analysis_id,
            interactions=interactions,
            overall_risk=overall_risk,
            confidence_score=confidence_score,
            processing_time=processing_time,
            recommendations=recommendations,
            uncertainty_assessment=uncertainty_assessment,
            metadata={
                "herbs_analyzed": len(herbs),
                "drugs_analyzed": len(drugs),
                "interactions_found": len(interactions),
                "patient_factors_considered": patient_factors is not None
            }
        )
    
    def _normalize_herbs(self, herbs: List[str]) -> List[str]:
        """Normalize herb names to standard forms."""
        normalized = []
        for herb in herbs:
            herb_lower = herb.lower().strip()
            found = False
            
            for standard_name, aliases in self.herb_aliases.items():
                if herb_lower in [alias.lower() for alias in aliases]:
                    normalized.append(standard_name)
                    found = True
                    break
            
            if not found:
                normalized.append(herb_lower)
        
        return normalized
    
    def _normalize_drugs(self, drugs: List[str]) -> List[str]:
        """Normalize drug names to standard forms."""
        normalized = []
        for drug in drugs:
            drug_lower = drug.lower().strip()
            found = False
            
            for standard_name, aliases in self.drug_aliases.items():
                if drug_lower in [alias.lower() for alias in aliases]:
                    normalized.append(standard_name)
                    found = True
                    break
            
            if not found:
                normalized.append(drug_lower)
        
        return normalized
    
    def _find_interactions(self, herbs: List[str], drugs: List[str]) -> List[Interaction]:
        """Find interactions between herbs and drugs."""
        interactions = []
        
        for herb in herbs:
            for drug in drugs:
                # Check direct interaction
                interaction_key = (herb, drug)
                if interaction_key in self.interactions_db:
                    interaction_data = self.interactions_db[interaction_key]
                    
                    interaction = Interaction(
                        herb=herb,
                        drug=drug,
                        severity=interaction_data["severity"],
                        description=interaction_data["description"],
                        mechanism=interaction_data["mechanism"],
                        evidence_level=interaction_data["evidence_level"],
                        confidence=interaction_data["confidence"],
                        recommendations=interaction_data["recommendations"]
                    )
                    interactions.append(interaction)
        
        return interactions
    
    def _calculate_overall_risk(self, interactions: List[Interaction]) -> str:
        """Calculate overall risk level."""
        if not interactions:
            return "low"
        
        severity_scores = {
            InteractionSeverity.LOW: 1,
            InteractionSeverity.MODERATE: 2,
            InteractionSeverity.HIGH: 3,
            InteractionSeverity.SEVERE: 4
        }
        
        max_severity = max(severity_scores[interaction.severity] for interaction in interactions)
        
        if max_severity >= 4:
            return "severe"
        elif max_severity >= 3:
            return "high"
        elif max_severity >= 2:
            return "moderate"
        else:
            return "low"
    
    def _calculate_confidence(self, interactions: List[Interaction], herb_count: int, drug_count: int) -> float:
        """Calculate confidence score for the analysis."""
        if not interactions:
            # High confidence in "no interactions" if we have good coverage
            base_confidence = 0.85 if herb_count <= 3 and drug_count <= 3 else 0.70
        else:
            # Average confidence of found interactions
            avg_confidence = sum(i.confidence for i in interactions) / len(interactions)
            base_confidence = avg_confidence
        
        # Adjust based on analysis complexity
        complexity_factor = min(1.0, 1.0 - (herb_count + drug_count - 2) * 0.05)
        final_confidence = base_confidence * complexity_factor
        
        return round(max(0.5, min(0.95, final_confidence)), 2)
    
    def _generate_recommendations(
        self,
        interactions: List[Interaction],
        patient_factors: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if not interactions:
            recommendations.append("No known significant interactions detected.")
            recommendations.append("Continue monitoring for any unexpected effects.")
        else:
            # Severity-based recommendations
            severe_interactions = [i for i in interactions if i.severity == InteractionSeverity.SEVERE]
            high_interactions = [i for i in interactions if i.severity == InteractionSeverity.HIGH]
            
            if severe_interactions:
                recommendations.append("URGENT: Severe interactions detected. Discontinue herbs immediately and consult healthcare provider.")
            elif high_interactions:
                recommendations.append("HIGH PRIORITY: Significant interactions found. Consult healthcare provider before continuing.")
            else:
                recommendations.append("Monitor closely for interaction symptoms.")
            
            # Specific recommendations from interactions
            for interaction in interactions:
                recommendations.extend(interaction.recommendations)
        
        # General safety recommendations
        recommendations.extend([
            "Always inform healthcare providers about all herbs and supplements being used.",
            "Start with lowest effective doses when combining herbs with medications.",
            "Monitor for any unusual symptoms or side effects."
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _assess_uncertainty(
        self,
        interactions: List[Interaction],
        herbs: List[str],
        drugs: List[str]
    ) -> Dict[str, Any]:
        """Assess uncertainty in the safety analysis."""
        # Calculate uncertainty based on evidence levels
        evidence_levels = [i.evidence_level for i in interactions]
        
        high_evidence = sum(1 for level in evidence_levels if level == "high")
        moderate_evidence = sum(1 for level in evidence_levels if level == "moderate")
        low_evidence = sum(1 for level in evidence_levels if level in ["low", "theoretical"])
        
        total_interactions = len(interactions)
        
        if total_interactions == 0:
            uncertainty_level = UncertaintyLevel.MODERATE
            uncertainty_score = 0.6
        elif high_evidence / total_interactions > 0.7:
            uncertainty_level = UncertaintyLevel.LOW
            uncertainty_score = 0.2
        elif moderate_evidence / total_interactions > 0.5:
            uncertainty_level = UncertaintyLevel.MODERATE
            uncertainty_score = 0.5
        else:
            uncertainty_level = UncertaintyLevel.HIGH
            uncertainty_score = 0.8
        
        return {
            "uncertainty_level": uncertainty_level.value,
            "uncertainty_score": uncertainty_score,
            "evidence_quality": {
                "high_evidence_interactions": high_evidence,
                "moderate_evidence_interactions": moderate_evidence,
                "low_evidence_interactions": low_evidence
            },
            "data_gaps": self._identify_data_gaps(herbs, drugs),
            "consultation_recommended": uncertainty_score > 0.6 or any(
                i.severity in [InteractionSeverity.HIGH, InteractionSeverity.SEVERE] 
                for i in interactions
            )
        }
    
    def _identify_data_gaps(self, herbs: List[str], drugs: List[str]) -> List[str]:
        """Identify gaps in interaction data."""
        gaps = []
        
        # Check for herbs/drugs not in our database
        known_herbs = set(self.herb_aliases.keys())
        known_drugs = set(self.drug_aliases.keys())
        
        for herb in herbs:
            if herb.lower() not in known_herbs:
                gaps.append(f"Limited data available for herb: {herb}")
        
        for drug in drugs:
            if drug.lower() not in known_drugs:
                gaps.append(f"Limited interaction data for drug: {drug}")
        
        return gaps
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "service_name": "OptimizedSafetyAnalyzer",
            "version": "1.0.0",
            "interaction_database_size": len(self.interactions_db),
            "supported_herbs": list(self.herb_aliases.keys()),
            "supported_drug_categories": list(self.drug_aliases.keys()),
            "severity_levels": [s.value for s in InteractionSeverity],
            "uncertainty_levels": [u.value for u in UncertaintyLevel]
        }

# Global instance
_global_safety_analyzer = None

def get_safety_analyzer() -> OptimizedSafetyAnalyzer:
    """Get or create global safety analyzer instance."""
    global _global_safety_analyzer
    if _global_safety_analyzer is None:
        _global_safety_analyzer = OptimizedSafetyAnalyzer()
    return _global_safety_analyzer