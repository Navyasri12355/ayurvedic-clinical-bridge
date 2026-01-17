#!/usr/bin/env python3
"""
Script to run the Ayurvedic Clinical Bridge API with authentication.
"""

import uvicorn
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("Starting Ayurvedic Clinical Bridge API with Authentication...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Frontend should connect to: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "ayurvedic_clinical_bridge.api.main_with_auth:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=True
    )