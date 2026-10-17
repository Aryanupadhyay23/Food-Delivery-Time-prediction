"""
Operational Geofencing & Hub Registry Database for 22 Indian Metros
"""

from typing import Dict, Any, List


class CityRegistry:
    """Provides operational geofencing and metadata for 22 Indian metro hubs."""

    DATABASE: Dict[str, Dict[str, Any]] = {
        "Bangalore": {
            "hub_code": "BANGRES19382",
            "city_type": "Metropolitian",
            "lat_range": (12.8000, 13.1500),
            "lng_range": (77.4500, 77.7800),
            "default_rest": (12.9716, 77.5946),
            "default_drop": (12.9352, 77.6245),
        },
        "Mumbai": {
            "hub_code": "MUMRES12345",
            "city_type": "Metropolitian",
            "lat_range": (18.8900, 19.3200),
            "lng_range": (72.7500, 73.0500),
            "default_rest": (19.0760, 72.8777),
            "default_drop": (19.1136, 72.8697),
        },
        "Hyderabad": {
            "hub_code": "HYDRES17294",
            "city_type": "Metropolitian",
            "lat_range": (17.2500, 17.5800),
            "lng_range": (78.2800, 78.6200),
            "default_rest": (17.3850, 78.4867),
            "default_drop": (17.4399, 78.3908),
        },
        "Pune": {
            "hub_code": "PUNERES18492",
            "city_type": "Urban",
            "lat_range": (18.4000, 18.6600),
            "lng_range": (73.7200, 74.0000),
            "default_rest": (18.5204, 73.8567),
            "default_drop": (18.5362, 73.8795),
        },
        "Kolkata": {
            "hub_code": "KOLRES14820",
            "city_type": "Metropolitian",
            "lat_range": (22.4000, 22.7500),
            "lng_range": (88.2500, 88.5000),
            "default_rest": (22.5726, 88.3639),
            "default_drop": (22.5850, 88.4100),
        },
        "Chennai": {
            "hub_code": "CHENRES15839",
            "city_type": "Metropolitian",
            "lat_range": (12.9000, 13.2500),
            "lng_range": (80.1000, 80.3500),
            "default_rest": (13.0827, 80.2707),
            "default_drop": (13.0012, 80.2565),
        },
        "Jaipur": {
            "hub_code": "JAPRES19302",
            "city_type": "Urban",
            "lat_range": (26.7500, 27.0500),
            "lng_range": (75.6800, 75.9200),
            "default_rest": (26.9124, 75.7873),
            "default_drop": (26.8520, 75.8050),
        },
        "Kochi": {
            "hub_code": "KOCRES16492",
            "city_type": "Urban",
            "lat_range": (9.8500, 10.1500),
            "lng_range": (76.2000, 76.4000),
            "default_rest": (9.9312, 76.2673),
            "default_drop": (9.9816, 76.2999),
        },
        "Goa": {
            "hub_code": "GOARES14920",
            "city_type": "Semi-Urban",
            "lat_range": (15.3500, 15.6500),
            "lng_range": (73.7000, 73.9800),
            "default_rest": (15.4989, 73.8278),
            "default_drop": (15.5180, 73.8500),
        },
        "Surat": {
            "hub_code": "SURRES11938",
            "city_type": "Urban",
            "lat_range": (21.0500, 21.2800),
            "lng_range": (72.7200, 72.9200),
            "default_rest": (21.1702, 72.8311),
            "default_drop": (21.1520, 72.7840),
        },
        "Indore": {
            "hub_code": "INDORES18492",
            "city_type": "Urban",
            "lat_range": (22.6200, 22.8200),
            "lng_range": (75.7800, 75.9500),
            "default_rest": (22.7196, 75.8577),
            "default_drop": (22.7533, 75.8937),
        },
        "Bhopal": {
            "hub_code": "BHPRES13940",
            "city_type": "Urban",
            "lat_range": (23.1500, 23.3500),
            "lng_range": (77.3000, 77.5200),
            "default_rest": (23.2599, 77.4126),
            "default_drop": (23.2120, 77.4360),
        },
        "Coimbatore": {
            "hub_code": "COIMBRES17492",
            "city_type": "Urban",
            "lat_range": (10.9000, 11.1200),
            "lng_range": (76.8800, 77.0800),
            "default_rest": (11.0168, 76.9558),
            "default_drop": (11.0280, 77.0020),
        },
        "Dehradun": {
            "hub_code": "DEHRES10293",
            "city_type": "Urban",
            "lat_range": (30.2200, 30.4200),
            "lng_range": (77.9500, 78.1200),
            "default_rest": (30.3165, 78.0322),
            "default_drop": (30.3540, 78.0680),
        },
        "Ludhiana": {
            "hub_code": "LUDHRES14820",
            "city_type": "Urban",
            "lat_range": (30.8200, 31.0000),
            "lng_range": (75.7500, 75.9500),
            "default_rest": (30.9010, 75.8573),
            "default_drop": (30.8850, 75.8120),
        },
        "Kanpur": {
            "hub_code": "KNPRES19382",
            "city_type": "Urban",
            "lat_range": (26.3500, 26.5500),
            "lng_range": (80.2500, 80.4200),
            "default_rest": (26.4499, 80.3319),
            "default_drop": (26.4820, 80.2980),
        },
        "Mysore": {
            "hub_code": "MYSRES18392",
            "city_type": "Urban",
            "lat_range": (12.2200, 12.3800),
            "lng_range": (76.5800, 76.7200),
            "default_rest": (12.2958, 76.6394),
            "default_drop": (12.3160, 76.6180),
        },
        "Ranchi": {
            "hub_code": "RANCHIRES12049",
            "city_type": "Urban",
            "lat_range": (23.2500, 23.4500),
            "lng_range": (85.2400, 85.3900),
            "default_rest": (23.3441, 85.3096),
            "default_drop": (23.3680, 85.3340),
        },
        "Aurangabad": {
            "hub_code": "AURGRES15920",
            "city_type": "Urban",
            "lat_range": (19.8000, 19.9500),
            "lng_range": (75.2500, 75.4200),
            "default_rest": (19.8762, 75.3433),
            "default_drop": (19.8820, 75.3780),
        },
        "Agra": {
            "hub_code": "AGRRES14920",
            "city_type": "Urban",
            "lat_range": (27.1000, 27.2500),
            "lng_range": (77.9200, 78.0800),
            "default_rest": (27.1767, 78.0081),
            "default_drop": (27.1580, 78.0420),
        },
        "Vadodara": {
            "hub_code": "VADRES18392",
            "city_type": "Urban",
            "lat_range": (22.2200, 22.3800),
            "lng_range": (73.1200, 73.2600),
            "default_rest": (22.3072, 73.1812),
            "default_drop": (22.3180, 73.1490),
        },
        "Prayagraj": {
            "hub_code": "ALHRES12940",
            "city_type": "Urban",
            "lat_range": (25.3800, 25.5200),
            "lng_range": (81.7800, 81.9200),
            "default_rest": (25.4358, 81.8463),
            "default_drop": (25.4560, 81.8290),
        }
    }

    TOP_HUBS: List[str] = ["Bangalore", "Mumbai", "Hyderabad", "Pune", "Kolkata", "Goa"]

    @classmethod
    def get_city_data(cls, city_name: str) -> Dict[str, Any]:
        """Returns metadata for the requested city or falls back to Bangalore."""
        return cls.DATABASE.get(city_name, cls.DATABASE["Bangalore"])

    @classmethod
    def get_all_city_names(cls) -> List[str]:
        """Returns all 22 registered Indian city names."""
        return list(cls.DATABASE.keys())

    @classmethod
    def is_within_bounds(cls, city_name: str, lat: float, lng: float) -> bool:
        """Verifies if a coordinate falls within the metropolitan bounding box."""
        city = cls.get_city_data(city_name)
        lat_min, lat_max = city["lat_range"]
        lng_min, lng_max = city["lng_range"]
        return (lat_min <= lat <= lat_max) and (lng_min <= lng <= lng_max)
