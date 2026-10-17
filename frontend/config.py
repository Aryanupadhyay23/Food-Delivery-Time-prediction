"""
Frontend Configuration & Environment Constants
"""

import os
from typing import Dict, Any

PAGE_CONFIG: Dict[str, Any] = {
    "page_title": "Food Delivery Time Prediction",
    "page_icon": "🛵",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8080").rstrip("/")
DEFAULT_CITY: str = "Bangalore"
