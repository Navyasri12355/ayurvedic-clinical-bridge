"""
Herb Synonym Matcher Stub
Provides herb name resolution and synonym matching.
"""

import logging

logger = logging.getLogger(__name__)


class HerbSynonymMatcher:
    """
    Stub implementation of HerbSynonymMatcher.
    Provides basic herb name resolution without complex synonym matching.
    """
    
    def __init__(self):
        """Initialize the synonym matcher."""
        logger.info("HerbSynonymMatcher initialized (stub implementation)")
    
    def resolve_herb_name(self, herb_name: str) -> str:
        """
        Resolve a herb name to its canonical form.
        Returns the input name as-is in this stub implementation.
        """
        return herb_name.title()
    
    def get_herb_suggestions(self, herb_name: str, max_suggestions: int = 3) -> list:
        """
        Get suggestions for a herb name.
        Returns empty list in this stub implementation.
        """
        return []
