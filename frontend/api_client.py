"""
HTTP API Client for FastAPI Prediction Service
"""

from typing import Dict, Any, Tuple, Optional
import requests

try:
    from frontend.config import BACKEND_URL
except ImportError:
    from config import BACKEND_URL


class DeliveryPredictionClient:
    """Encapsulates HTTP requests and error handling for the inference backend."""

    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url

    def predict_delivery_time(
        self,
        payload: Dict[str, Any],
        timeout_seconds: int = 10
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Sends prediction payload to the FastAPI backend service.
        Returns:
            (success: bool, response_dict: Optional[dict], error_message: str)
        """
        url = f"{self.base_url}/predict"
        try:
            response = requests.post(url, json=payload, timeout=timeout_seconds)
            if response.status_code == 200:
                return True, response.json(), ""
            else:
                return False, None, f"Prediction service error (HTTP {response.status_code}): {response.text}"
        except requests.exceptions.ConnectionError:
            return False, None, "Delivery Prediction Service Offline. Please verify that the backend API is running on port 8080."
        except requests.exceptions.Timeout:
            return False, None, "Prediction request timed out after 10 seconds. The server might be overloaded."
        except Exception as exc:
            return False, None, f"Unexpected error during prediction request: {str(exc)}"
