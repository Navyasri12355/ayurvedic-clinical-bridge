"""
LLM-Enhanced API Endpoints for Ayurvedic Clinical Bridge

Provides Flask routes for:
- LLM-powered clinical reasoning
- Conversational queries with memory
- Enhanced response generation
- Safety analysis
- Explanation generation
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import logging
import os
from datetime import datetime

from .llm_service import LLMService, LLMConfig, LLMProvider
from .clinical_reasoning_engine import ClinicalReasoningEngine

logger = logging.getLogger(__name__)

# Create blueprint for LLM routes
llm_bp = Blueprint("llm", __name__, url_prefix="/api/llm")

# Global instances (initialize on app startup)
llm_service = None
reasoning_engine = None


def init_llm_service(app=None):
    """Initialize LLM service from environment variables"""
    global llm_service, reasoning_engine

    # Get configuration from environment
    provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()
    model_name = os.getenv("LLM_MODEL", "mistral")
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # Map provider string to enum
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "anthropic": LLMProvider.ANTHROPIC,
        "ollama": LLMProvider.OLLAMA,
        "groq": LLMProvider.GROQ,
    }

    provider = provider_map.get(provider_name, LLMProvider.OLLAMA)

    try:
        config = LLMConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_service = LLMService.initialize(config)
        reasoning_engine = ClinicalReasoningEngine()

        if app:
            app.logger.info(
                f"✓ LLM Service initialized: {provider_name} - {model_name}"
            )
        return True

    except Exception as e:
        if app:
            app.logger.error(f"✗ Failed to initialize LLM service: {e}")
        logger.error(f"Failed to initialize LLM service: {e}")
        return False


def require_llm(f):
    """Decorator to ensure LLM service is available"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not llm_service or not llm_service.is_available():
            return (
                jsonify(
                    {
                        "error": "LLM service not available",
                        "message": "Please configure LLM provider",
                    }
                ),
                503,
            )
        return f(*args, **kwargs)

    return decorated_function


@llm_bp.route("/health", methods=["GET"])
def llm_health():
    """Check LLM service health"""
    if llm_service and llm_service.is_available():
        return jsonify(
            {
                "status": "healthy",
                "llm_service": "active",
                "timestamp": datetime.now().isoformat(),
            }
        )
    else:
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "llm_service": "inactive",
                    "message": "LLM service not configured",
                }
            ),
            503,
        )


@llm_bp.route("/clinical-analysis", methods=["POST"])
@require_llm
def clinical_analysis():
    """
    Perform LLM-enhanced clinical analysis.

    Request body:
    {
        "query": "Patient with fever and cough",
        "entities": {"symptoms": [...], "diseases": [...]},
        "predictions": {"disease_predictions": [...], "herbs": [...]},
        "user_id": "user123"
    }
    """
    try:
        data = request.json
        query = data.get("query")
        entities = data.get("entities", {})
        predictions = data.get("predictions", {})
        user_id = data.get("user_id", "default")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        result = reasoning_engine.analyze_clinical_query(
            query=query,
            detected_entities=entities,
            model_predictions=predictions,
            user_id=user_id,
        )

        return jsonify(
            {
                "status": "success",
                "analysis": {
                    "clinical_summary": result.clinical_summary,
                    "key_findings": result.key_findings,
                    "recommendations": result.recommendations,
                    "safety_concerns": result.safety_concerns,
                    "follow_up_questions": result.follow_up_questions,
                    "confidence_level": result.confidence_level,
                    "reasoning_chain": result.reasoning_chain,
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Clinical analysis error: {e}")
        return jsonify({"error": str(e), "type": "clinical_analysis_error"}), 500


@llm_bp.route("/generate-response", methods=["POST"])
@require_llm
def generate_response():
    """
    Generate natural language response with LLM.

    Request body:
    {
        "query": "What herbs help with digestion?",
        "entities": {...},
        "herbs": [{...}],
        "disease_predictions": [{...}],
        "safety_alerts": ["Monitor if on blood thinners"],
        "user_id": "user123",
        "user_role": "general_user"
    }
    """
    try:
        data = request.json
        query = data.get("query")
        entities = data.get("entities", {})
        herbs = data.get("herbs", [])
        diseases = data.get("disease_predictions", [])
        safety_alerts = data.get("safety_alerts", [])
        user_id = data.get("user_id", "default")
        user_role = data.get("user_role", "general_user")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        response = reasoning_engine.generate_response(
            query=query,
            detected_entities=entities,
            herb_data=herbs,
            disease_predictions=diseases,
            safety_alerts=safety_alerts,
            user_id=user_id,
            user_role=user_role,
        )

        return jsonify(
            {
                "status": "success",
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return jsonify({"error": str(e), "type": "response_generation_error"}), 500


@llm_bp.route("/safety-analysis", methods=["POST"])
@require_llm
def safety_analysis():
    """
    Perform LLM-enhanced safety analysis.

    Request body:
    {
        "query": "Taking aspirin and ginger together",
        "entities": {"herbs": [...], "medications": [...]},
        "interactions": [{...}],
        "user_id": "user123"
    }
    """
    try:
        data = request.json
        query = data.get("query")
        entities = data.get("entities", {})
        interactions = data.get("interactions", [])
        user_id = data.get("user_id", "default")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        analysis = reasoning_engine.analyze_safety(
            query=query,
            detected_entities=entities,
            potential_interactions=interactions,
            user_id=user_id,
        )

        return jsonify(
            {
                "status": "success",
                "safety_analysis": analysis,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Safety analysis error: {e}")
        return jsonify({"error": str(e), "type": "safety_analysis_error"}), 500


@llm_bp.route("/chat", methods=["POST"])
@require_llm
def conversational_chat():
    """
    Handle conversational queries with memory.

    Request body:
    {
        "message": "Tell me more about that interaction",
        "user_id": "user123"
    }
    """
    try:
        data = request.json
        message = data.get("message")
        user_id = data.get("user_id", "default")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        response = reasoning_engine.conversational_followup(
            user_message=message,
            user_id=user_id,
        )

        return jsonify(
            {
                "status": "success",
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e), "type": "chat_error"}), 500


@llm_bp.route("/explain-prediction", methods=["POST"])
@require_llm
def explain_prediction():
    """
    Generate explanation for model predictions.

    Request body:
    {
        "prediction": {"disease": "Pitta Imbalance", "confidence": 0.92},
        "query": "I have burning sensation",
        "model_name": "BioBERT",
        "user_id": "user123"
    }
    """
    try:
        data = request.json
        prediction = data.get("prediction")
        query = data.get("query")
        model_name = data.get("model_name", "Unknown")
        user_id = data.get("user_id", "default")

        if not prediction or not query:
            return jsonify({"error": "prediction and query are required"}), 400

        explanation = reasoning_engine.explain_prediction(
            prediction=prediction,
            query=query,
            model_name=model_name,
            user_id=user_id,
        )

        return jsonify(
            {
                "status": "success",
                "explanation": explanation,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Explanation generation error: {e}")
        return jsonify({"error": str(e), "type": "explanation_error"}), 500


@llm_bp.route("/conversation/<user_id>/summary", methods=["GET"])
def conversation_summary(user_id):
    """Get conversation summary for user"""
    try:
        summary = reasoning_engine.get_conversation_summary(user_id)
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        logger.error(f"Conversation summary error: {e}")
        return jsonify({"error": str(e)}), 500


@llm_bp.route("/conversation/<user_id>/clear", methods=["DELETE"])
def clear_conversation(user_id):
    """Clear conversation history for user"""
    try:
        reasoning_engine.clear_conversation(user_id)
        return jsonify(
            {"status": "success", "message": f"Conversation cleared for {user_id}"}
        )
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        return jsonify({"error": str(e)}), 500


@llm_bp.route("/capabilities", methods=["GET"])
@require_llm
def llm_capabilities():
    """Get LLM service capabilities"""
    return jsonify(
        {
            "status": "available",
            "capabilities": {
                "clinical_analysis": {
                    "description": "Multi-step clinical analysis with reasoning",
                    "endpoint": "/api/llm/clinical-analysis",
                },
                "generate_response": {
                    "description": "Natural language response generation",
                    "endpoint": "/api/llm/generate-response",
                },
                "safety_analysis": {
                    "description": "LLM-enhanced drug interaction analysis",
                    "endpoint": "/api/llm/safety-analysis",
                },
                "chat": {
                    "description": "Conversational queries with memory",
                    "endpoint": "/api/llm/chat",
                },
                "explain_prediction": {
                    "description": "Explain model predictions",
                    "endpoint": "/api/llm/explain-prediction",
                },
                "conversation_management": {
                    "description": "Get/clear conversation history",
                    "endpoints": {
                        "summary": "/api/llm/conversation/<user_id>/summary",
                        "clear": "/api/llm/conversation/<user_id>/clear",
                    },
                },
            },
        }
    )
