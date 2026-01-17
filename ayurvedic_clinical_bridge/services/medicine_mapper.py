"""
Medicine Mapping Service for Ayurvedic Clinical Bridge

This service provides mapping between allopathic and ayurvedic medicines
using the ayurgenix dataset and additional medical knowledge.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MedicineMapping:
    """Data class for medicine mapping information."""
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

@dataclass
class AyurvedicRecommendation:
    """Data class for ayurvedic medicine recommendations."""
    herb_name: str
    dosage: str
    formulation: str
    preparation_method: str
    timing: str
    duration: str
    precautions: str

class MedicineMapper:
    """Service for mapping allopathic medicines to ayurvedic alternatives."""
    
    def __init__(self):
        self.dataset_path = Path("data/ayurgenix_dataset.csv")
        self.processed_dataset_path = Path("data/datasets/ayurgenixai_processed.csv")
        self.medicine_data = None
        self.allopathic_mappings = {}
        self._load_datasets()
        self._initialize_mappings()
    
    def _load_datasets(self):
        """Load the ayurgenix datasets."""
        try:
            # Try processed dataset first
            if self.processed_dataset_path.exists():
                self.medicine_data = pd.read_csv(self.processed_dataset_path)
                logger.info(f"Loaded processed ayurgenix dataset with {len(self.medicine_data)} records")
                logger.info(f"Dataset columns: {list(self.medicine_data.columns)}")
            elif self.dataset_path.exists():
                self.medicine_data = pd.read_csv(self.dataset_path)
                logger.info(f"Loaded ayurgenix dataset with {len(self.medicine_data)} records")
                logger.info(f"Dataset columns: {list(self.medicine_data.columns)}")
            else:
                logger.warning(f"No dataset found at {self.dataset_path} or {self.processed_dataset_path}")
                self.medicine_data = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.medicine_data = pd.DataFrame()
    
    def _initialize_mappings(self):
        """Initialize common allopathic to ayurvedic medicine mappings."""
        # Common allopathic medicines and their ayurvedic alternatives
        self.allopathic_mappings = {
            # Diabetes medications
            "metformin": {
                "ayurvedic_alternatives": ["Jamun", "Gudmar", "Fenugreek", "Bitter Melon"],
                "primary_herb": "Jamun",
                "dosage": "3g daily",
                "formulation": "Powder with warm water",
                "dosha": "Kapha-Pitta",
                "safety_notes": "Monitor blood sugar levels closely",
                "contraindications": "Pregnancy, severe hypoglycemia"
            },
            "insulin": {
                "ayurvedic_alternatives": ["Gudmar", "Jamun", "Karela", "Fenugreek"],
                "primary_herb": "Gudmar",
                "dosage": "2g twice daily",
                "formulation": "Capsules or powder",
                "dosha": "Kapha",
                "safety_notes": "Requires medical supervision",
                "contraindications": "Type 1 diabetes, severe complications"
            },
            
            # Hypertension medications
            "amlodipine": {
                "ayurvedic_alternatives": ["Arjuna", "Ashwagandha", "Garlic", "Hibiscus"],
                "primary_herb": "Arjuna",
                "dosage": "5g daily",
                "formulation": "Bark powder with warm water",
                "dosha": "Pitta-Vata",
                "safety_notes": "Monitor blood pressure regularly",
                "contraindications": "Severe heart conditions, pregnancy"
            },
            "lisinopril": {
                "ayurvedic_alternatives": ["Ashwagandha", "Arjuna", "Brahmi"],
                "primary_herb": "Ashwagandha",
                "dosage": "3g daily",
                "formulation": "Root powder with milk",
                "dosha": "Vata-Pitta",
                "safety_notes": "May interact with other medications",
                "contraindications": "Autoimmune conditions"
            },
            
            # Pain medications
            "ibuprofen": {
                "ayurvedic_alternatives": ["Turmeric", "Boswellia", "Ginger", "Ashwagandha"],
                "primary_herb": "Turmeric",
                "dosage": "1-2g daily",
                "formulation": "Powder with warm milk",
                "dosha": "Pitta-Kapha",
                "safety_notes": "May cause stomach irritation in high doses",
                "contraindications": "Gallstones, blood thinning medications"
            },
            "acetaminophen": {
                "ayurvedic_alternatives": ["Willow Bark", "Turmeric", "Ginger"],
                "primary_herb": "Turmeric",
                "dosage": "1g twice daily",
                "formulation": "Capsules or powder",
                "dosha": "Pitta",
                "safety_notes": "Generally safe in recommended doses",
                "contraindications": "Liver disease"
            },
            
            # Respiratory medications
            "albuterol": {
                "ayurvedic_alternatives": ["Tulsi", "Mulethi", "Vasaka", "Ginger"],
                "primary_herb": "Tulsi",
                "dosage": "5-10 leaves daily",
                "formulation": "Fresh leaves or tea",
                "dosha": "Kapha-Vata",
                "safety_notes": "Not for acute asthma attacks",
                "contraindications": "Severe respiratory distress"
            },
            "prednisone": {
                "ayurvedic_alternatives": ["Ashwagandha", "Turmeric", "Boswellia"],
                "primary_herb": "Ashwagandha",
                "dosage": "3-5g daily",
                "formulation": "Root powder",
                "dosha": "Vata-Pitta",
                "safety_notes": "Gradual tapering recommended",
                "contraindications": "Autoimmune conditions"
            },
            
            # Digestive medications
            "omeprazole": {
                "ayurvedic_alternatives": ["Amla", "Aloe Vera", "Licorice", "Fennel"],
                "primary_herb": "Amla",
                "dosage": "10ml juice daily",
                "formulation": "Fresh juice or powder",
                "dosha": "Pitta",
                "safety_notes": "May increase stomach acid initially",
                "contraindications": "Severe GERD"
            },
            "simethicone": {
                "ayurvedic_alternatives": ["Ajwain", "Fennel", "Ginger", "Hing"],
                "primary_herb": "Ajwain",
                "dosage": "1 tsp with warm water",
                "formulation": "Seeds or powder",
                "dosha": "Vata-Kapha",
                "safety_notes": "Generally safe",
                "contraindications": "None known"
            },
            
            # Anxiety/Sleep medications
            "lorazepam": {
                "ayurvedic_alternatives": ["Ashwagandha", "Brahmi", "Jatamansi", "Shankhpushpi"],
                "primary_herb": "Ashwagandha",
                "dosage": "3g at bedtime",
                "formulation": "Powder with warm milk",
                "dosha": "Vata-Pitta",
                "safety_notes": "May cause drowsiness",
                "contraindications": "Pregnancy, severe depression"
            },
            "zolpidem": {
                "ayurvedic_alternatives": ["Jatamansi", "Brahmi", "Ashwagandha"],
                "primary_herb": "Jatamansi",
                "dosage": "2g at bedtime",
                "formulation": "Powder with warm water",
                "dosha": "Vata",
                "safety_notes": "Natural sleep aid",
                "contraindications": "Severe depression"
            },
            
            # Antibiotics (supportive herbs)
            "amoxicillin": {
                "ayurvedic_alternatives": ["Neem", "Turmeric", "Tulsi", "Garlic"],
                "primary_herb": "Neem",
                "dosage": "2-3g daily",
                "formulation": "Leaf powder or capsules",
                "dosha": "Pitta-Kapha",
                "safety_notes": "Supportive only, not replacement for antibiotics",
                "contraindications": "Pregnancy, autoimmune conditions"
            },
            
            # Cholesterol medications
            "atorvastatin": {
                "ayurvedic_alternatives": ["Arjuna", "Guggul", "Garlic", "Fenugreek"],
                "primary_herb": "Guggul",
                "dosage": "500mg twice daily",
                "formulation": "Standardized extract",
                "dosha": "Kapha",
                "safety_notes": "Monitor liver function",
                "contraindications": "Liver disease, pregnancy"
            }
        }
    
    def find_ayurvedic_alternative(self, allopathic_medicine: str, disease: Optional[str] = None) -> Optional[MedicineMapping]:
        """
        Find ayurvedic alternatives for a given allopathic medicine.
        
        Args:
            allopathic_medicine: Name of the allopathic medicine
            disease: Optional disease context for better matching
            
        Returns:
            MedicineMapping object with ayurvedic alternatives
        """
        medicine_lower = allopathic_medicine.lower().strip()
        
        # Check direct mappings first
        if medicine_lower in self.allopathic_mappings:
            mapping_data = self.allopathic_mappings[medicine_lower]
            
            # Get disease-specific information from dataset if available
            disease_info = self._get_disease_info(disease) if disease else {}
            
            return MedicineMapping(
                allopathic_medicine=allopathic_medicine,
                ayurvedic_alternatives=mapping_data["ayurvedic_alternatives"],
                disease=disease or "General",
                dosage=mapping_data["dosage"],
                formulation=mapping_data["formulation"],
                dosha=mapping_data["dosha"],
                constitution=disease_info.get("constitution", mapping_data["dosha"]),
                confidence_score=0.85,  # High confidence for direct mappings
                safety_notes=mapping_data["safety_notes"],
                contraindications=mapping_data["contraindications"]
            )
        
        # Try fuzzy matching for similar medicine names
        return self._fuzzy_match_medicine(medicine_lower, disease)
    
    def _fuzzy_match_medicine(self, medicine: str, disease: Optional[str] = None) -> Optional[MedicineMapping]:
        """Attempt fuzzy matching for medicine names."""
        # Common medicine name patterns and their mappings
        patterns = {
            r".*metformin.*": "metformin",
            r".*insulin.*": "insulin",
            r".*amlodipine.*": "amlodipine",
            r".*ibuprofen.*": "ibuprofen",
            r".*acetaminophen.*|.*paracetamol.*": "acetaminophen",
            r".*omeprazole.*": "omeprazole",
            r".*ashwagandha.*": "ashwagandha_herb",
            r".*turmeric.*|.*curcumin.*": "turmeric_herb"
        }
        
        for pattern, mapped_medicine in patterns.items():
            if re.match(pattern, medicine, re.IGNORECASE):
                if mapped_medicine in self.allopathic_mappings:
                    return self.find_ayurvedic_alternative(mapped_medicine, disease)
        
        return None
    
    def _get_disease_info(self, disease: str) -> Dict:
        """Get disease-specific information from the dataset."""
        if self.medicine_data.empty or not disease:
            return {}
        
        # Find matching disease in dataset
        disease_matches = self.medicine_data[
            self.medicine_data['Disease'].str.contains(disease, case=False, na=False)
        ]
        
        if not disease_matches.empty:
            row = disease_matches.iloc[0]
            return {
                "constitution": row.get("Constitution/Prakriti", ""),
                "doshas": row.get("Doshas", ""),
                "ayurvedic_herbs": row.get("Ayurvedic Herbs", ""),
                "formulation": row.get("Formulation", ""),
                "diet_recommendations": row.get("Diet and Lifestyle Recommendations", "")
            }
        
        return {}
    
    def get_disease_based_recommendations(self, disease: str) -> List[AyurvedicRecommendation]:
        """
        Get ayurvedic medicine recommendations based on disease.
        
        Args:
            disease: Name of the disease
            
        Returns:
            List of AyurvedicRecommendation objects
        """
        # Always use comprehensive recommendations for better results
        return self._get_comprehensive_disease_recommendations(disease)
    
    def _get_comprehensive_disease_recommendations(self, disease: str) -> List[AyurvedicRecommendation]:
        """Get comprehensive Ayurvedic recommendations for a disease."""
        disease_lower = disease.lower().strip()
        recommendations = []
        
        logger.info(f"Getting comprehensive recommendations for disease: '{disease_lower}'")
        
        # Comprehensive disease-herb mappings with detailed information
        disease_recommendations = {
            "arthritis": [
                {
                    "herb_name": "Guggulu (Commiphora mukul)",
                    "dosage": "500mg twice daily",
                    "formulation": "Standardized extract capsules",
                    "preparation_method": "Take with warm water after meals",
                    "timing": "Morning and evening after food",
                    "duration": "3-6 months or as directed by practitioner",
                    "precautions": "Avoid during pregnancy. Monitor if on blood thinners."
                },
                {
                    "herb_name": "Shallaki (Boswellia serrata)",
                    "dosage": "300-400mg twice daily",
                    "formulation": "Standardized extract tablets",
                    "preparation_method": "Take with warm milk or water",
                    "timing": "30 minutes before meals",
                    "duration": "2-4 months for optimal results",
                    "precautions": "Generally safe. Consult if on anti-inflammatory drugs."
                },
                {
                    "herb_name": "Rasna (Pluchea lanceolata)",
                    "dosage": "3-5g powder daily",
                    "formulation": "Root powder or decoction",
                    "preparation_method": "Mix powder in warm water or prepare decoction",
                    "timing": "Twice daily before meals",
                    "duration": "1-3 months",
                    "precautions": "Start with lower dose. Avoid in severe kidney disease."
                }
            ],
            "diabetes": [
                {
                    "herb_name": "Gudmar (Gymnema sylvestre)",
                    "dosage": "400-500mg twice daily",
                    "formulation": "Leaf extract capsules",
                    "preparation_method": "Take with water before meals",
                    "timing": "30 minutes before breakfast and dinner",
                    "duration": "3-6 months with regular monitoring",
                    "precautions": "Monitor blood sugar levels. Adjust medication as needed."
                },
                {
                    "herb_name": "Jamun (Syzygium cumini)",
                    "dosage": "2-3g seed powder daily",
                    "formulation": "Dried seed powder",
                    "preparation_method": "Mix in water or buttermilk",
                    "timing": "Empty stomach in morning",
                    "duration": "2-4 months",
                    "precautions": "May lower blood sugar significantly. Monitor closely."
                },
                {
                    "herb_name": "Karela (Momordica charantia)",
                    "dosage": "30ml fresh juice daily",
                    "formulation": "Fresh juice or dried extract",
                    "preparation_method": "Fresh juice on empty stomach or extract with water",
                    "timing": "Early morning before breakfast",
                    "duration": "Ongoing with breaks",
                    "precautions": "May cause stomach upset initially. Start with small amounts."
                }
            ],
            "hypertension": [
                {
                    "herb_name": "Arjuna (Terminalia arjuna)",
                    "dosage": "500mg twice daily",
                    "formulation": "Bark powder or extract",
                    "preparation_method": "Decoction with milk or water",
                    "timing": "Morning and evening",
                    "duration": "3-6 months",
                    "precautions": "Monitor blood pressure regularly. Safe for long-term use."
                },
                {
                    "herb_name": "Punarnava (Boerhavia diffusa)",
                    "dosage": "3-5g powder daily",
                    "formulation": "Root powder or extract",
                    "preparation_method": "Mix in warm water or honey",
                    "timing": "Twice daily before meals",
                    "duration": "2-3 months",
                    "precautions": "Ensure adequate hydration. Avoid in severe kidney disease."
                }
            ],
            "asthma": [
                {
                    "herb_name": "Vasaka (Adhatoda vasica)",
                    "dosage": "5-10ml leaf juice twice daily",
                    "formulation": "Fresh leaf juice or syrup",
                    "preparation_method": "Fresh juice with honey or prepared syrup",
                    "timing": "Morning and evening",
                    "duration": "1-3 months",
                    "precautions": "May cause mild nausea initially. Reduce dose if needed."
                },
                {
                    "herb_name": "Pushkarmool (Inula racemosa)",
                    "dosage": "1-2g powder twice daily",
                    "formulation": "Root powder",
                    "preparation_method": "Mix with honey or warm water",
                    "timing": "After meals",
                    "duration": "2-4 months",
                    "precautions": "Start with lower dose. Avoid during acute attacks."
                }
            ],
            "migraine": [
                {
                    "herb_name": "Brahmi (Bacopa monnieri)",
                    "dosage": "300-500mg daily",
                    "formulation": "Standardized extract",
                    "preparation_method": "Take with water or milk",
                    "timing": "Morning with breakfast",
                    "duration": "2-3 months",
                    "precautions": "May cause mild drowsiness initially. Take with food."
                },
                {
                    "herb_name": "Jatamansi (Nardostachys jatamansi)",
                    "dosage": "1-2g powder daily",
                    "formulation": "Root powder or extract",
                    "preparation_method": "Mix with warm milk before bed",
                    "timing": "Evening before sleep",
                    "duration": "1-2 months",
                    "precautions": "May enhance sleep. Avoid driving after taking."
                }
            ],
            "gastritis": [
                {
                    "herb_name": "Amalaki (Emblica officinalis)",
                    "dosage": "10ml fresh juice daily",
                    "formulation": "Fresh juice or powder",
                    "preparation_method": "Fresh juice on empty stomach or powder with water",
                    "timing": "Early morning before breakfast",
                    "duration": "2-3 months",
                    "precautions": "May increase stomach acid initially in some people."
                },
                {
                    "herb_name": "Yashtimadhu (Glycyrrhiza glabra)",
                    "dosage": "1-2g powder twice daily",
                    "formulation": "Root powder or extract",
                    "preparation_method": "Mix with warm milk or water",
                    "timing": "After meals",
                    "duration": "1-2 months",
                    "precautions": "Avoid in high blood pressure. Monitor potassium levels."
                }
            ],
            "insomnia": [
                {
                    "herb_name": "Ashwagandha (Withania somnifera)",
                    "dosage": "3-5g powder daily",
                    "formulation": "Root powder",
                    "preparation_method": "Mix with warm milk and honey",
                    "timing": "1 hour before bedtime",
                    "duration": "2-3 months",
                    "precautions": "May cause drowsiness. Avoid during pregnancy."
                },
                {
                    "herb_name": "Jatamansi (Nardostachys jatamansi)",
                    "dosage": "1-2g powder daily",
                    "formulation": "Root powder",
                    "preparation_method": "Mix with warm milk",
                    "timing": "30 minutes before sleep",
                    "duration": "1-2 months",
                    "precautions": "Natural sedative. Avoid driving after taking."
                }
            ],
            "anxiety": [
                {
                    "herb_name": "Brahmi (Bacopa monnieri)",
                    "dosage": "300-500mg twice daily",
                    "formulation": "Standardized extract",
                    "preparation_method": "Take with water after meals",
                    "timing": "Morning and evening",
                    "duration": "2-4 months",
                    "precautions": "May cause mild drowsiness initially."
                },
                {
                    "herb_name": "Shankhpushpi (Convolvulus pluricaulis)",
                    "dosage": "3-5g powder daily",
                    "formulation": "Whole plant powder",
                    "preparation_method": "Mix with warm milk or water",
                    "timing": "Morning and evening",
                    "duration": "2-3 months",
                    "precautions": "Generally safe. Start with lower dose."
                }
            ]
        }
        
        # Add more diseases as needed
        matched_disease = None
        for disease_key in disease_recommendations:
            if disease_key in disease_lower:
                matched_disease = disease_key
                logger.info(f"Found match for '{disease_lower}' with key '{disease_key}'")
                for rec_data in disease_recommendations[disease_key]:
                    recommendations.append(AyurvedicRecommendation(**rec_data))
                break
        
        if not matched_disease:
            logger.info(f"No direct match found for '{disease_lower}', checking general recommendations")
        
        # If no specific recommendations found, provide general recommendations
        if not recommendations and disease_lower in ["gastritis", "insomnia", "anxiety", "depression", "obesity", "anemia", "bronchitis", "constipation", "diarrhea", "eczema"]:
            logger.info(f"Getting general recommendations for '{disease_lower}'")
            recommendations = self._get_general_recommendations_for_disease(disease_lower)
        
        logger.info(f"Returning {len(recommendations)} recommendations for '{disease_lower}'")
        return recommendations
    
    def _get_general_recommendations_for_disease(self, disease: str) -> List[AyurvedicRecommendation]:
        """Get general recommendations for diseases not in the detailed mapping."""
        herbs = self._get_herbs_for_disease(disease.title()).split(", ")
        recommendations = []
        
        for herb in herbs[:3]:  # Limit to top 3 herbs
            herb = herb.strip()
            if herb:
                recommendations.append(AyurvedicRecommendation(
                    herb_name=herb,
                    dosage=self._get_dosage_for_herb(herb, disease),
                    formulation="Powder, capsules, or as directed",
                    preparation_method="As per traditional preparation methods",
                    timing="As directed by Ayurvedic practitioner",
                    duration="2-3 months or as advised",
                    precautions="Consult qualified Ayurvedic practitioner before use"
                ))
        
        return recommendations
    
    def _get_dosage_for_herb(self, herb: str, disease: str) -> str:
        """Get appropriate dosage for herb based on disease."""
        herb_lower = herb.lower()
        
        dosage_map = {
            "guggulu": "500mg twice daily",
            "shallaki": "300-400mg twice daily", 
            "rasna": "3-5g powder daily",
            "gudmar": "400-500mg twice daily",
            "jamun": "2-3g seed powder daily",
            "karela": "30ml fresh juice daily",
            "arjuna": "500mg twice daily",
            "punarnava": "3-5g powder daily",
            "vasaka": "5-10ml juice twice daily",
            "brahmi": "300-500mg daily",
            "turmeric": "500mg with black pepper daily",
            "ashwagandha": "300-500mg twice daily",
            "triphala": "3-5g powder daily"
        }
        
        for herb_key in dosage_map:
            if herb_key in herb_lower:
                return dosage_map[herb_key]
        
        return "As per practitioner guidance"
    
    def _get_formulation_for_herb(self, herb: str) -> str:
        """Get appropriate formulation for herb."""
        herb_lower = herb.lower()
        
        if any(word in herb_lower for word in ["powder", "churna"]):
            return "Powder form"
        elif any(word in herb_lower for word in ["extract", "capsule"]):
            return "Standardized extract"
        elif "juice" in herb_lower:
            return "Fresh juice"
        elif "oil" in herb_lower:
            return "Medicated oil"
        else:
            return "Powder or extract as available"
    
    def _get_timing_for_herb(self, herb: str, disease: str) -> str:
        """Get appropriate timing for herb based on disease."""
        herb_lower = herb.lower()
        
        if "brahmi" in herb_lower or "shankhpushpi" in herb_lower:
            return "Morning with breakfast"
        elif "jatamansi" in herb_lower or "ashwagandha" in herb_lower:
            return "Evening before sleep"
        elif "gudmar" in herb_lower or "karela" in herb_lower:
            return "Before meals"
        else:
            return "As directed by practitioner"
    
    def _get_duration_for_disease(self, disease: str) -> str:
        """Get treatment duration for disease."""
        chronic_conditions = ["arthritis", "diabetes", "hypertension", "asthma"]
        
        if any(condition in disease.lower() for condition in chronic_conditions):
            return "3-6 months with regular monitoring"
        else:
            return "1-3 months or until symptoms improve"
    
    def _get_precautions_for_herb(self, herb: str, disease: str) -> str:
        """Get precautions for herb based on disease."""
        herb_lower = herb.lower()
        
        precautions_map = {
            "guggulu": "Avoid during pregnancy. Monitor if on blood thinners.",
            "gudmar": "Monitor blood sugar levels closely. Adjust medications as needed.",
            "arjuna": "Generally safe for long-term use. Monitor blood pressure.",
            "ashwagandha": "Avoid during pregnancy. May enhance sleep.",
            "brahmi": "May cause mild drowsiness initially. Take with food."
        }
        
        for herb_key in precautions_map:
            if herb_key in herb_lower:
                return precautions_map[herb_key]
        
        return "Consult qualified Ayurvedic practitioner before use"
    
    def _extract_dosage(self, formulation: str, herb: str) -> str:
        """Extract dosage information from formulation text."""
        if not formulation or formulation == "nan":
            return "As per practitioner guidance"
        
        # Look for dosage patterns in formulation
        dosage_patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:g|gm|gram)",
            r"(\d+(?:\.\d+)?)\s*(?:mg|milligram)",
            r"(\d+(?:\.\d+)?)\s*(?:tsp|teaspoon)",
            r"(\d+(?:\.\d+)?)\s*(?:tbsp|tablespoon)",
            r"(\d+(?:\.\d+)?)\s*(?:ml|milliliter)"
        ]
        
        for pattern in dosage_patterns:
            match = re.search(pattern, formulation, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {match.group(0).split()[-1]}"
        
        return "As per practitioner guidance"
    
    def _get_preparation_method(self, herb: str) -> str:
        """Get preparation method for specific herbs."""
        preparation_methods = {
            "turmeric": "Mix powder with warm milk or water",
            "ginger": "Fresh juice or powder with warm water",
            "tulsi": "Fresh leaves or dried powder as tea",
            "ashwagandha": "Root powder with warm milk at bedtime",
            "amla": "Fresh juice on empty stomach",
            "neem": "Leaf powder with water",
            "arjuna": "Bark powder with warm water",
            "brahmi": "Leaf powder with ghee or warm water",
            "jamun": "Seed powder with water before meals",
            "gudmar": "Leaf powder with warm water before meals"
        }
        
        herb_lower = herb.lower()
        for key, method in preparation_methods.items():
            if key in herb_lower:
                return method
        
        return "Consult Ayurvedic practitioner for preparation method"
    
    def get_interaction_warnings(self, allopathic_medicine: str, ayurvedic_herbs: List[str]) -> List[str]:
        """
        Get potential interaction warnings between allopathic medicines and ayurvedic herbs.
        
        Args:
            allopathic_medicine: Name of allopathic medicine
            ayurvedic_herbs: List of ayurvedic herbs
            
        Returns:
            List of warning messages
        """
        warnings = []
        medicine_lower = allopathic_medicine.lower()
        
        # Known interactions
        interactions = {
            "warfarin": ["turmeric", "garlic", "ginger"],
            "insulin": ["gudmar", "jamun", "fenugreek"],
            "metformin": ["gudmar", "bitter melon"],
            "digoxin": ["licorice"],
            "lithium": ["ashwagandha"],
            "immunosuppressants": ["ashwagandha", "tulsi"]
        }
        
        for medicine, interacting_herbs in interactions.items():
            if medicine in medicine_lower:
                for herb in ayurvedic_herbs:
                    herb_lower = herb.lower()
                    for interacting_herb in interacting_herbs:
                        if interacting_herb in herb_lower:
                            warnings.append(
                                f"Potential interaction: {herb} may interact with {allopathic_medicine}. "
                                f"Monitor closely and consult healthcare provider."
                            )
        
        return warnings
    
    def search_by_symptoms(self, symptoms: List[str]) -> List[Dict]:
        """
        Search for ayurvedic recommendations based on symptoms.
        
        Args:
            symptoms: List of symptoms
            
        Returns:
            List of dictionaries with disease and herb recommendations
        """
        if self.medicine_data.empty:
            # Return mock data if dataset is not available
            return self._get_mock_symptom_results(symptoms)
        
        try:
            results = []
            
            # Check what columns are available
            available_columns = list(self.medicine_data.columns)
            logger.info(f"Available columns in dataset: {available_columns}")
            
            # Determine symptom columns based on what's available
            symptom_columns = []
            if 'Symptom1' in available_columns:
                symptom_columns.append('Symptom1')
            if 'Symptom2' in available_columns:
                symptom_columns.append('Symptom2')
            if 'Symptoms' in available_columns:
                symptom_columns.append('Symptoms')
            
            if not symptom_columns:
                logger.warning("No symptom columns found in dataset, using mock data")
                return self._get_mock_symptom_results(symptoms)
            
            for symptom in symptoms:
                # Search in available symptom columns
                conditions = []
                for col in symptom_columns:
                    conditions.append(
                        self.medicine_data[col].str.contains(symptom, case=False, na=False)
                    )
                
                # Combine conditions with OR
                if len(conditions) == 1:
                    symptom_matches = self.medicine_data[conditions[0]]
                else:
                    combined_condition = conditions[0]
                    for condition in conditions[1:]:
                        combined_condition = combined_condition | condition
                    symptom_matches = self.medicine_data[combined_condition]
                
                for _, row in symptom_matches.iterrows():
                    # Get disease from available columns
                    disease = row.get("Disease", row.get("disease", "Unknown"))
                    
                    # Get dosha from available columns
                    dosha = (row.get("Parameter1", "") or 
                            row.get("Dosha", "") or 
                            row.get("Doshas", "") or
                            self._get_dosha_for_disease(disease))
                    
                    # Combine symptoms from available columns
                    symptoms_text = []
                    for col in symptom_columns:
                        if col in row and pd.notna(row[col]):
                            symptoms_text.append(str(row[col]))
                    combined_symptoms = ", ".join(symptoms_text) if symptoms_text else symptom
                    
                    # Get ayurvedic herbs for this disease
                    ayurvedic_herbs = self._get_herbs_for_disease(disease)
                    
                    results.append({
                        "disease": disease,
                        "symptoms": combined_symptoms,
                        "ayurvedic_herbs": ayurvedic_herbs,
                        "formulation": self._get_formulation_for_disease(disease),
                        "dosha": dosha,
                        "constitution": self._get_constitution_for_dosha(dosha),
                        "diet_recommendations": self._get_diet_recommendations_for_disease(disease)
                    })
            
            # Remove duplicates based on disease
            unique_results = []
            seen_diseases = set()
            for result in results:
                if result["disease"] not in seen_diseases:
                    unique_results.append(result)
                    seen_diseases.add(result["disease"])
            
            return unique_results if unique_results else self._get_mock_symptom_results(symptoms)
            
        except Exception as e:
            logger.error(f"Error in search_by_symptoms: {e}")
            # Fallback to mock data on any error
            return self._get_mock_symptom_results(symptoms)
    
    def _get_mock_symptom_results(self, symptoms: List[str]) -> List[Dict]:
        """Generate mock symptom search results when dataset is not available."""
        mock_results = []
        
        # Common symptom-disease mappings
        symptom_disease_map = {
            "fatigue": {"disease": "Diabetes", "herbs": "Gudmar, Jamun, Karela"},
            "headache": {"disease": "Migraine", "herbs": "Brahmi, Jatamansi, Shankhpushpi"},
            "pain": {"disease": "Arthritis", "herbs": "Guggulu, Shallaki, Rasna"},
            "dizziness": {"disease": "Hypertension", "herbs": "Arjuna, Punarnava, Brahmi"},
            "inflammation": {"disease": "Arthritis", "herbs": "Turmeric, Guggulu, Shallaki"},
            "stiffness": {"disease": "Arthritis", "herbs": "Guggulu, Rasna, Nirgundi"},
            "weakness": {"disease": "Anemia", "herbs": "Punarnava, Amalaki, Mandur Bhasma"},
            "burning": {"disease": "Gastritis", "herbs": "Amalaki, Yashtimadhu, Shatavari"},
            "nausea": {"disease": "Gastritis", "herbs": "Ginger, Cardamom, Fennel"},
            "swelling": {"disease": "Obesity", "herbs": "Guggulu, Triphala, Punarnava"}
        }
        
        for symptom in symptoms:
            symptom_lower = symptom.lower().strip()
            if symptom_lower in symptom_disease_map:
                disease_info = symptom_disease_map[symptom_lower]
                mock_results.append({
                    "disease": disease_info["disease"],
                    "symptoms": symptom,
                    "ayurvedic_herbs": disease_info["herbs"],
                    "formulation": "Powder or capsules with warm water",
                    "dosha": self._get_dosha_for_disease(disease_info["disease"]),
                    "constitution": "Consult Ayurvedic practitioner",
                    "diet_recommendations": self._get_diet_recommendations_for_disease(disease_info["disease"])
                })
        
        return mock_results
    
    def _get_herbs_for_disease(self, disease: str) -> str:
        """Get ayurvedic herbs for a specific disease."""
        disease_herbs = {
            "Diabetes": "Gudmar, Jamun, Karela, Fenugreek",
            "Hypertension": "Arjuna, Punarnava, Brahmi, Jatamansi",
            "Arthritis": "Guggulu, Shallaki, Rasna, Nirgundi",
            "Asthma": "Vasaka, Pushkarmool, Kantakari, Bharangi",
            "Migraine": "Brahmi, Jatamansi, Shankhpushpi, Saraswatarishta",
            "Gastritis": "Amalaki, Yashtimadhu, Shatavari, Kamadudha",
            "Insomnia": "Ashwagandha, Brahmi, Jatamansi, Shankhpushpi",
            "Anxiety": "Ashwagandha, Brahmi, Shankhpushpi, Saraswatarishta",
            "Depression": "Brahmi, Mandukaparni, Saraswatarishta, Medhya Rasayana",
            "Obesity": "Guggulu, Triphala, Vrikshamla, Punarnava",
            "Anemia": "Punarnava, Amalaki, Mandur Bhasma, Lohasava",
            "Bronchitis": "Vasaka, Kantakari, Pushkarmool, Bharangi",
            "Constipation": "Triphala, Isabgol, Castor oil, Haritaki",
            "Diarrhea": "Kutaja, Bilva, Musta, Dadimashtak",
            "Eczema": "Neem, Manjistha, Turmeric, Khadira"
        }
        return disease_herbs.get(disease, "Consult Ayurvedic practitioner")
    
    def _get_formulation_for_disease(self, disease: str) -> str:
        """Get recommended formulation for a disease."""
        return "Powder with warm water or as directed by practitioner"
    
    def _get_constitution_for_dosha(self, dosha: str) -> str:
        """Get constitution recommendation based on dosha."""
        if "Vata" in dosha:
            return "Vata constitution - needs grounding and nourishment"
        elif "Pitta" in dosha:
            return "Pitta constitution - needs cooling and calming"
        elif "Kapha" in dosha:
            return "Kapha constitution - needs stimulation and lightness"
        return "Mixed constitution - consult practitioner"
    
    def _get_dosha_for_disease(self, disease: str) -> str:
        """Get primary dosha involvement for a disease."""
        disease_dosha = {
            "Diabetes": "Kapha-Vata",
            "Hypertension": "Vata-Pitta",
            "Arthritis": "Vata",
            "Asthma": "Kapha-Vata",
            "Migraine": "Vata-Pitta",
            "Gastritis": "Pitta",
            "Insomnia": "Vata",
            "Anxiety": "Vata",
            "Depression": "Kapha-Tamas",
            "Obesity": "Kapha",
            "Anemia": "Pitta",
            "Bronchitis": "Kapha",
            "Constipation": "Vata",
            "Diarrhea": "Pitta",
            "Eczema": "Pitta-Kapha"
        }
        return disease_dosha.get(disease, "Tridosha")
    
    def _get_diet_recommendations_for_disease(self, disease: str) -> str:
        """Get dietary recommendations for a disease."""
        diet_recommendations = {
            "Diabetes": "Low sugar, high fiber diet with bitter vegetables",
            "Hypertension": "Low salt, cooling foods, avoid spicy items",
            "Arthritis": "Anti-inflammatory diet, warm foods, avoid cold items",
            "Asthma": "Warm, light foods, avoid cold and heavy items",
            "Migraine": "Regular meals, avoid trigger foods, cooling diet",
            "Gastritis": "Cooling, alkaline foods, avoid spicy and acidic items",
            "Insomnia": "Light dinner, warm milk, avoid caffeine",
            "Anxiety": "Nourishing, grounding foods, regular meal times",
            "Depression": "Energizing foods, avoid heavy and cold items",
            "Obesity": "Light, warm foods with bitter and pungent tastes",
            "Anemia": "Iron-rich foods, vitamin C sources, avoid tea with meals",
            "Bronchitis": "Warm, light foods, avoid cold and dairy",
            "Constipation": "High fiber, adequate water, healthy fats",
            "Diarrhea": "Light, easily digestible foods, adequate fluids",
            "Eczema": "Cooling foods, avoid spicy, sour, and fermented items"
        }
        return diet_recommendations.get(disease, "Balanced diet as per constitution")

# Global instance
medicine_mapper = MedicineMapper()

def get_medicine_mapper() -> MedicineMapper:
    """Get the global medicine mapper instance."""
    return medicine_mapper