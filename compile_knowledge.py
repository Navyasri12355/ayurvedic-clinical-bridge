#!/usr/bin/env python3
"""
Knowledge Compilation Script for Ayurvedic Clinical Bridge

This script compiles all knowledge sources into a single JSON file
for efficient learning and retrieval.
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ayurvedic_clinical_bridge.services.knowledge_compiler import KnowledgeCompiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to compile knowledge sources."""
    logger.info("Starting knowledge compilation process...")
    
    try:
        # Initialize knowledge compiler
        compiler = KnowledgeCompiler()
        
        # Compile all knowledge sources
        result = compiler.compile_all_knowledge_sources()
        
        if result.success:
            logger.info("✅ Knowledge compilation completed successfully!")
            logger.info(f"📊 Statistics:")
            logger.info(f"   - Total concepts: {result.total_concepts}")
            logger.info(f"   - Total relationships: {result.total_relationships}")
            logger.info(f"   - Compilation time: {result.compilation_time:.2f}s")
            logger.info(f"   - Output file: {result.output_file}")
            logger.info(f"   - File size: {result.file_size_mb:.2f} MB")
            
            if result.warnings:
                logger.warning("⚠️  Warnings:")
                for warning in result.warnings:
                    logger.warning(f"   - {warning}")
                    
            return 0
        else:
            logger.error("❌ Knowledge compilation failed!")
            if result.errors:
                logger.error("Errors:")
                for error in result.errors:
                    logger.error(f"   - {error}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Knowledge compilation failed with exception: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)