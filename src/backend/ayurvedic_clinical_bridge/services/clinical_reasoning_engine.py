"""
Clinical Reasoning Engine for Ayurvedic Clinical Bridge

Combines BiLSTM-CRF, BioBERT transformers with LLM for advanced clinical analysis,
reasoning, and multi-turn conversations.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .llm_service import (
    LLMService,
    CLINICAL_REASONING_SYSTEM_PROMPT,
    RESPONSE_GENERATION_SYSTEM_PROMPT,
    SAFETY_ANALYSIS_SYSTEM_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a message in conversation history"""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str = None
    entities: Optional[Dict[str, Any]] = None  # Detected entities in user message

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for LLM API"""
        return {"role": self.role, "content": self.content}


@dataclass
class ClinicalAnalysisResult:
    """Result of clinical analysis"""

    clinical_summary: str
    key_findings: List[str]
    recommendations: List[str]
    safety_concerns: List[str]
    follow_up_questions: List[str]
    confidence_level: str  # "high", "moderate", "low"
    reasoning_chain: str  # Step-by-step reasoning


class ConversationContext:
    """Manages conversation history and context for a user session"""

    def __init__(self, user_id: str, max_history: int = 20):
        self.user_id = user_id
        self.messages: List[ConversationMessage] = []
        self.max_history = max_history
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "user_profile": {},  # Store user preferences, conditions, etc.
        }

    def add_message(self, message: ConversationMessage):
        """Add message to conversation history"""
        self.messages.append(message)
        # Keep only recent messages to avoid context window overflow
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]

    def get_conversation_for_llm(self) -> List[Dict[str, str]]:
        """Get conversation in format suitable for LLM API"""
        return [msg.to_dict() for msg in self.messages]

    def get_context_summary(self) -> str:
        """Get summary of conversation context"""
        if not self.messages:
            return "No previous context"

        summary_parts = []
        for msg in self.messages[-5:]:  # Last 5 messages
            role = "User" if msg.role == "user" else "Assistant"
            summary_parts.append(f"{role}: {msg.content[:100]}...")

        return "\n".join(summary_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation"""
        return {
            "user_id": self.user_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                }
                for msg in self.messages
            ],
            "metadata": self.metadata,
        }


class ClinicalReasoningEngine:
    """
    Advanced clinical reasoning engine that combines:
    - BiLSTM-CRF NER for entity extraction
    - BioBERT for semantic understanding
    - LLM for reasoning and explanation
    """

    def __init__(self):
        self.llm = LLMService()
        self.conversations: Dict[str, ConversationContext] = {}

    def _get_conversation(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for user"""
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationContext(user_id)
        return self.conversations[user_id]

    def analyze_clinical_query(
        self,
        query: str,
        detected_entities: Dict[str, Any],
        model_predictions: Dict[str, Any],
        user_id: str = "default",
    ) -> ClinicalAnalysisResult:
        """
        Perform multi-step clinical analysis using LLM.

        Args:
            query: Original user query
            detected_entities: Entities from NER models
            model_predictions: Predictions from BioBERT/BiLSTM
            user_id: User session ID

        Returns:
            ClinicalAnalysisResult with comprehensive analysis
        """
        conversation = self._get_conversation(user_id)

        # Build analysis prompt with context
        analysis_prompt = self._build_clinical_analysis_prompt(
            query, detected_entities, model_predictions, conversation
        )

        try:
            response = self.llm.generate(
                prompt=analysis_prompt,
                system_prompt=CLINICAL_REASONING_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=2000,
            )

            # Parse LLM response
            result = self._parse_clinical_response(response)

            # Store in conversation history
            user_msg = ConversationMessage(
                role="user", content=query, entities=detected_entities
            )
            assistant_msg = ConversationMessage(role="assistant", content=response)

            conversation.add_message(user_msg)
            conversation.add_message(assistant_msg)

            return result

        except Exception as e:
            logger.error(f"Clinical analysis error: {e}")
            raise

    def generate_response(
        self,
        query: str,
        detected_entities: Dict[str, Any],
        herb_data: List[Dict[str, Any]],
        disease_predictions: List[Dict[str, Any]],
        safety_alerts: List[str],
        user_id: str = "default",
        user_role: str = "general_user",
    ) -> str:
        """
        Generate natural language response using LLM.

        Args:
            query: Original user query
            detected_entities: Extracted entities
            herb_data: Recommended herbs
            disease_predictions: Disease predictions
            safety_alerts: Safety concerns
            user_id: User session ID
            user_role: "general_user" or "qualified_practitioner"

        Returns:
            Natural language response
        """
        conversation = self._get_conversation(user_id)

        # Build response prompt
        response_prompt = self._build_response_generation_prompt(
            query,
            detected_entities,
            herb_data,
            disease_predictions,
            safety_alerts,
            user_role,
        )

        try:
            llm_response = self.llm.generate(
                prompt=response_prompt,
                system_prompt=RESPONSE_GENERATION_SYSTEM_PROMPT,
                temperature=0.8,
                max_tokens=1500,
            )

            # Store in conversation
            user_msg = ConversationMessage(role="user", content=query)
            assistant_msg = ConversationMessage(role="assistant", content=llm_response)

            conversation.add_message(user_msg)
            conversation.add_message(assistant_msg)

            return llm_response

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            raise

    def analyze_safety(
        self,
        query: str,
        detected_entities: Dict[str, Any],
        potential_interactions: List[Dict[str, Any]],
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Perform LLM-enhanced safety analysis.

        Args:
            query: Original query
            detected_entities: Detected entities
            potential_interactions: Herb-drug interactions from models
            user_id: User session ID

        Returns:
            Detailed safety analysis
        """

        safety_prompt = self._build_safety_analysis_prompt(
            query, detected_entities, potential_interactions
        )

        try:
            response = self.llm.generate(
                prompt=safety_prompt,
                system_prompt=SAFETY_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.6,  # Lower temperature for safety
                max_tokens=1500,
            )

            return {
                "analysis": response,
                "timestamp": datetime.now().isoformat(),
                "severity_level": self._extract_severity_level(response),
            }

        except Exception as e:
            logger.error(f"Safety analysis error: {e}")
            raise

    def conversational_followup(
        self,
        user_message: str,
        user_id: str,
    ) -> str:
        """
        Handle follow-up questions in conversation with memory.

        Args:
            user_message: User's follow-up question
            user_id: User session ID

        Returns:
            LLM response maintaining conversation context
        """
        conversation = self._get_conversation(user_id)

        # Add user message
        conversation.add_message(ConversationMessage(role="user", content=user_message))

        # Get conversation messages for LLM
        messages = conversation.get_conversation_for_llm()

        try:
            response = self.llm.generate_with_context(
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )

            # Add assistant response
            conversation.add_message(
                ConversationMessage(role="assistant", content=response)
            )

            return response

        except Exception as e:
            logger.error(f"Conversation followup error: {e}")
            raise

    def explain_prediction(
        self,
        prediction: Dict[str, Any],
        query: str,
        model_name: str,
        user_id: str = "default",
    ) -> str:
        """
        Generate human-readable explanation for model predictions.

        Args:
            prediction: Model prediction
            query: Original query
            model_name: Name of model
            user_id: User session ID

        Returns:
            Explanation text
        """
        explanation_prompt = self._build_explanation_prompt(
            prediction, query, model_name
        )

        try:
            response = self.llm.generate(
                prompt=explanation_prompt,
                system_prompt=EXPLANATION_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1000,
            )

            return response

        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            raise

    # ==================== HELPER METHODS ====================

    def _build_clinical_analysis_prompt(
        self,
        query: str,
        entities: Dict[str, Any],
        predictions: Dict[str, Any],
        context: ConversationContext,
    ) -> str:
        """Build prompt for clinical analysis"""
        context_text = (
            f"Previous conversation context:\n{context.get_context_summary()}\n\n"
            if context.messages
            else ""
        )

        entities_text = json.dumps(entities, indent=2) if entities else "None detected"
        predictions_text = json.dumps(predictions, indent=2) if predictions else "None"

        prompt = f"""Clinical Query Analysis Request:

{context_text}
Current Query: {query}

Detected Entities:
{entities_text}

Model Predictions:
{predictions_text}

Please provide:
1. Clinical Summary: Brief overview of findings
2. Key Findings: Main clinical insights
3. Recommendations: Actionable treatment suggestions
4. Safety Concerns: Any risks or contraindications
5. Follow-up Questions: What should be explored further
6. Reasoning Chain: Step-by-step clinical reasoning
7. Confidence Level: High/Moderate/Low for your assessment

Format your response as structured JSON."""

        return prompt

    def _build_response_generation_prompt(
        self,
        query: str,
        entities: Dict[str, Any],
        herbs: List[Dict[str, Any]],
        diseases: List[Dict[str, Any]],
        safety_alerts: List[str],
        user_role: str,
    ) -> str:
        """Build prompt for natural language response"""
        herbs_text = json.dumps(herbs[:5], indent=2) if herbs else "None recommended"
        diseases_text = (
            json.dumps(diseases[:3], indent=2) if diseases else "No predictions"
        )
        safety_text = (
            "\n".join(f"• {alert}" for alert in safety_alerts)
            if safety_alerts
            else "No safety concerns"
        )

        tone = (
            "technical and comprehensive"
            if user_role == "qualified_practitioner"
            else "friendly and accessible"
        )

        prompt = f"""Generate a helpful Ayurvedic health response.

User Query: {query}
User Role: {user_role}
Tone: {tone}

Detected Elements:
{json.dumps(entities, indent=2) if entities else "None"}

Recommended Herbs:
{herbs_text}

Disease Predictions:
{diseases_text}

Safety Alerts:
{safety_text}

Create a response that:
- Answers the user's question clearly
- Explains relevant Ayurvedic concepts
- Provides herb recommendations with dosage (if applicable)
- Includes safety considerations
- Recommends consulting healthcare providers for serious conditions

Generate a natural, well-structured response."""

        return prompt

    def _build_safety_analysis_prompt(
        self,
        query: str,
        entities: Dict[str, Any],
        interactions: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for safety analysis"""
        interactions_text = (
            json.dumps(interactions, indent=2) if interactions else "No interactions detected"
        )

        prompt = f"""Perform comprehensive safety analysis for Ayurvedic recommendation.

Query: {query}

Detected Herbs/Medications:
{json.dumps(entities, indent=2) if entities else "None"}

Potential Interactions:
{interactions_text}

Analyze:
1. Severity Level: Mild / Moderate / Severe
2. Mechanism: How do these interact?
3. Clinical Relevance: Is this significant?
4. Recommendations: Avoid / Monitor / Adjust dose / Consider alternatives
5. Evidence Quality: Strong / Moderate / Limited
6. Patient-Friendly Explanation: Simple terms for patient understanding

Format as structured analysis."""

        return prompt

    def _build_explanation_prompt(
        self,
        prediction: Dict[str, Any],
        query: str,
        model_name: str,
    ) -> str:
        """Build prompt for explanation generation"""
        prompt = f"""Explain this medical prediction in clear, understandable terms.

Model: {model_name}
Original Query: {query}

Prediction:
{json.dumps(prediction, indent=2)}

Provide:
1. What the model predicted and why
2. Key factors that influenced this prediction
3. Relevance to Ayurvedic principles
4. Confidence in this prediction
5. Important limitations or caveats
6. Suggested next steps

Make it understandable to a general audience while maintaining accuracy."""

        return prompt

    def _parse_clinical_response(self, response: str) -> ClinicalAnalysisResult:
        """Parse LLM response into structured result"""
        try:
            # Try to extract JSON if present
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                data = json.loads(json_str)
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                data = json.loads(response[json_start:json_end])
            else:
                # Fallback: create structured result from text
                data = {
                    "clinical_summary": response[:200],
                    "key_findings": ["See full analysis"],
                    "recommendations": ["Consult healthcare provider"],
                    "safety_concerns": [],
                    "follow_up_questions": [],
                    "confidence_level": "moderate",
                    "reasoning_chain": response,
                }

            return ClinicalAnalysisResult(
                clinical_summary=data.get("clinical_summary", ""),
                key_findings=data.get("key_findings", []),
                recommendations=data.get("recommendations", []),
                safety_concerns=data.get("safety_concerns", []),
                follow_up_questions=data.get("follow_up_questions", []),
                confidence_level=data.get("confidence_level", "moderate"),
                reasoning_chain=data.get("reasoning_chain", response),
            )
        except Exception as e:
            logger.warning(f"Failed to parse clinical response: {e}")
            return ClinicalAnalysisResult(
                clinical_summary=response,
                key_findings=[],
                recommendations=[],
                safety_concerns=[],
                follow_up_questions=[],
                confidence_level="low",
                reasoning_chain=response,
            )

    def _extract_severity_level(self, response: str) -> str:
        """Extract severity level from safety analysis"""
        response_lower = response.lower()
        if "severe" in response_lower:
            return "severe"
        elif "moderate" in response_lower:
            return "moderate"
        elif "mild" in response_lower:
            return "mild"
        else:
            return "unknown"

    def clear_conversation(self, user_id: str):
        """Clear conversation history for user"""
        if user_id in self.conversations:
            del self.conversations[user_id]
            logger.info(f"Cleared conversation history for user {user_id}")

    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        if user_id not in self.conversations:
            return {"messages": 0, "context": "No conversation"}

        conv = self.conversations[user_id]
        return {
            "user_id": user_id,
            "total_messages": len(conv.messages),
            "context_summary": conv.get_context_summary(),
            "metadata": conv.metadata,
        }
