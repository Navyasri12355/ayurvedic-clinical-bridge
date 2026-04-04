"""
Explanation Manager Service

Manages explanation state to prevent duplication and ensure a single
explanation per query via LRU caching.

No domain-specific keyword lists or hard-coded Ayurvedic term checks
are used — all branching is driven by fields already present in the
model output.
"""

import hashlib
import time
from typing import Dict, Any, Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class ExplanationManager:
    """
    Cache-backed manager that ensures a single explanation is generated
    per unique (query, context) pair.
    """

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._active: set[str] = set()
        logger.info("ExplanationManager initialised (size=%d, ttl=%ds)", cache_size, cache_ttl)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _query_id(self, query_text: str, user_context: Optional[Dict] = None) -> str:
        raw = query_text.lower().strip()
        if user_context:
            raw += str(user_context.get('role', '')) + str(user_context.get('preferences', ''))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _is_valid(self, qid: str) -> bool:
        return (
            qid in self._timestamps
            and (time.time() - self._timestamps[qid]) < self.cache_ttl
        )

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if now - ts >= self.cache_ttl]
        for k in expired:
            self._cache.pop(k, None)
            self._timestamps.pop(k, None)

    def _trim(self):
        while len(self._cache) > self.cache_size:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest, None)
            self._timestamps.pop(oldest, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_explanation(
        self,
        query_text: str,
        explainability_service,
        user_context: Optional[Dict] = None,
        main_prediction: Optional[Dict] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        qid = self._query_id(query_text, user_context)
        self._evict_expired()

        if not force_refresh and qid in self._cache and self._is_valid(qid):
            # LRU: move to end
            entry = self._cache.pop(qid)
            self._cache[qid] = entry
            logger.debug("Cache hit for qid=%s", qid)
            return entry

        if qid in self._active:
            return {'available': False, 'status': 'generating',
                    'reason': 'Explanation is currently being generated'}

        try:
            self._active.add(qid)
            result = self._generate(query_text, explainability_service, main_prediction)
            self._cache[qid] = result
            self._timestamps[qid] = time.time()
            self._trim()
            return result
        except Exception as e:
            logger.error("Explanation generation error for qid=%s: %s", qid, e)
            return {'available': False, 'status': 'error', 'reason': str(e)}
        finally:
            self._active.discard(qid)

    # ------------------------------------------------------------------
    # Generation — purely model-driven
    # ------------------------------------------------------------------

    def _generate(
        self,
        query_text: str,
        explainability_service,
        main_prediction: Optional[Dict],
    ) -> Dict[str, Any]:
        if not explainability_service:
            return {'available': False, 'status': 'unavailable',
                    'reason': 'Explainability service not available'}

        try:
            explanation = explainability_service.explain_prediction(query_text, top_k=3)
        except Exception as e:
            logger.error("explain_prediction raised: %s", e)
            return {'available': False, 'status': 'error', 'reason': str(e)}

        if not explanation.get('explanation_available', False):
            return {'available': False, 'status': 'failed',
                    'reason': 'Model explanation generation failed'}

        # Build summary using main_prediction if supplied (keeps the
        # displayed prediction and the explanation text consistent),
        # otherwise fall back to whatever the explainability service produced.
        if main_prediction:
            disease = main_prediction.get('disease', 'Unknown')
            conf_pct = main_prediction.get('confidence_percentage', '0%')
            word_exps = explanation.get('word_explanations', [])
            positive_words = [w['word'] for w in word_exps[:3] if w.get('contribution') == 'positive']
            summary = f"The AI model predicted '{disease}' with {conf_pct} confidence."
            if positive_words:
                summary += f" Key influencing words: {', '.join(positive_words)}."
            summary += " This shows which words most influenced the AI's decision."
        else:
            summary = explanation.get('summary', '')

        return {
            'available': True,
            'status': 'success',
            'method': explanation.get('explanation_method', 'SHAP'),
            'feature_importance': explanation.get('feature_importance', {}),
            'explanation_summary': summary,
            'word_importance': explanation.get('word_explanations', [])[:5],
            'confidence_assessment': explanation.get('confidence_assessment', {}),
            'visualization_data': explanation.get('visualization_data', {}),
            'user_summary': summary,
        }

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate_explanations(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple explanation keys in a response into a single 'explanation' key."""
        keys = ['explanation', 'ai_explanation', 'shap_explanation', 'explainability']
        found = [k for k in keys if response_data.get(k)]
        if len(found) <= 1:
            return response_data

        primary = response_data[found[0]]
        for k in found[1:]:
            other = response_data.pop(k, {})
            if isinstance(other, dict) and isinstance(primary, dict):
                for field in ('feature_importance', 'explanation_summary'):
                    if field in other and field not in primary:
                        primary[field] = other[field]

        response_data['explanation'] = primary
        return response_data

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def clear_cache(self):
        self._cache.clear()
        self._timestamps.clear()
        self._active.clear()
        logger.info("Explanation cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
            'active_generations': len(self._active),
            'cache_ttl': self.cache_ttl,
            'oldest_entry_age': (
                min(time.time() - ts for ts in self._timestamps.values())
                if self._timestamps else 0
            ),
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_explanation_manager: ExplanationManager | None = None


def get_explanation_manager() -> ExplanationManager:
    global _explanation_manager
    if _explanation_manager is None:
        _explanation_manager = ExplanationManager()
    return _explanation_manager