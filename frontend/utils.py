"""
Geospatial Math & Temporal Domain Utilities
"""

import math
from datetime import datetime, timedelta
from typing import Tuple


class GeoMathUtility:
    """Computes great-circle distances and realistic transit road paths."""

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates distance in kilometers between two GPS points using the Haversine formula."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = abs(lat1), abs(lon1), abs(lat2), abs(lon2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
        return round(R * c, 2)

    @staticmethod
    def estimate_road_distance(straight_line_km: float, winding_factor: float = 1.35) -> float:
        """Estimates actual surface road driving distance with urban winding factor."""
        return round(straight_line_km * winding_factor, 1)


class TimeWindowUtility:
    """Formats and projects delivery arrival windows based on dispatch timestamps."""

    @staticmethod
    def compute_arrival_window(time_ordered_str: str, eta_minutes: int, city_name: str) -> str:
        """Computes a doorstep delivery timestamp window (e.g. 01:15 PM - 01:25 PM)."""
        try:
            h, m = map(int, time_ordered_str.split(":"))
            order_dt = datetime.now().replace(hour=h, minute=m)
            arr_start = (order_dt + timedelta(minutes=eta_minutes - 2)).strftime("%I:%M %p")
            arr_end = (order_dt + timedelta(minutes=eta_minutes + 3)).strftime("%I:%M %p")
            return f"Estimated Doorstep Window in <b>{city_name}</b>: <b>{arr_start} – {arr_end}</b>"
        except Exception:
            return f"Estimated Transit Time in <b>{city_name}</b>: ~<b>{eta_minutes} mins</b>"

    @staticmethod
    def classify_speed_tier(eta_minutes: int) -> Tuple[str, str]:
        """Classifies delivery speed into Express, Standard, or Congested tiers."""
        if eta_minutes < 25:
            return "pill-express", "EXPRESS (< 25 MINS)"
        elif eta_minutes <= 40:
            return "pill-standard", "ON-SCHEDULE (25 – 40 MINS)"
        else:
            return "pill-delay", "CONGESTION DELAY (> 40 MINS)"
