import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FoodDeliveryFeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Stateless: No parameters needed
        pass

    def fit(self, X, y=None):
        # Stateless: Nothing to learn
        return self

    def transform(self, X):
        df = X.copy()
        
    
        # 1. Feature Extraction

        
        # City Code
        if "Delivery_person_ID" in df.columns:
            df["city_code"] = df["Delivery_person_ID"].astype(str).str.split("RES").str[0]

        # Date Features
        if "Order_Date" in df.columns:
            # We assume Pydantic provides a valid date string
            df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
            
            # Extract features
            # NOTE: Model specifically needs 'day_name' (e.g., 'monday', 'sunday')
            df["day_name"] = df["Order_Date"].dt.day_name().str.lower()
            
            # These might be dropped later depending on training logic, but we generate them safely
            df["day"] = df["Order_Date"].dt.day
            df["month"] = df["Order_Date"].dt.month
            df["is_weekend"] = df["Order_Date"].dt.dayofweek.isin([5, 6]).astype(int)

        # Time Features
        if "Time_Orderd" in df.columns:
            order_time = df["Time_Orderd"].apply(self._decimal_to_hhmm)
            df["time_of_day"] = order_time.apply(self._time_of_day)


        # 2. Categorical Standardization & Creation
  
        
        # Create 'city_type' from 'City' before we drop 'City'
        # The model expects 'city_type', not 'City'
        if "City" in df.columns:
            df["city_type"] = df["City"].astype(str).str.strip().str.lower()

        # Lowercase and strip text for other categoricals
        cat_cols = ["Weather_conditions", "Road_traffic_density", "Type_of_order", "Festival"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        
        # 3. Distance Calculation (Haversine)
        
        loc_cols = ["Restaurant_latitude", "Restaurant_longitude", 
                    "Delivery_location_latitude", "Delivery_location_longitude"]
        
        if all(c in df.columns for c in loc_cols):
            R = 6371.0 # Earth radius in km
            
            # Pydantic guarantees these are valid Indian coordinates
            lat1 = np.radians(df["Restaurant_latitude"])
            lon1 = np.radians(df["Restaurant_longitude"])
            lat2 = np.radians(df["Delivery_location_latitude"])
            lon2 = np.radians(df["Delivery_location_longitude"])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            df["distance"] = R * c

        
        # 4. Final Cleanup (Renaming & Dropping)
  
        # Rename to match model's expected feature names
        rename_map = {
            "Delivery_person_Age": "rider_age",
            "Delivery_person_Ratings": "rider_ratings",
            "Weather_conditions": "weather",
            "Road_traffic_density": "traffic_density",
            "Vehicle_condition": "vehicle_condition",
            "Type_of_order": "order_type",
            "Type_of_vehicle": "vehicle_type",
            "Festival": "festival",
            "multiple_deliveries": "multiple_deliveries"
        }
        df = df.rename(columns=rename_map)

        # Drop unused columns
        cols_to_drop = [
            "ID", "Delivery_person_ID", "rider_id", 
            "Time_Orderd", "Time_Order_picked", 
            "City", "city_name", 
            "Restaurant_latitude", "Restaurant_longitude", 
            "Delivery_location_latitude", "Delivery_location_longitude",
            "Order_Date", "order_date", "order_time", "distance_bin", "prep_time_minutes",
            "day", "month", "is_weekend" 
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        return df

    # Utils
    def _decimal_to_hhmm(self, x):
        """Standardizes time format."""
        if pd.isna(x) or str(x).strip() == "": return np.nan
        x = str(x).strip()
        if ":" in x:
            try:
                h, m = map(int, x.split(":"))
                return f"{h%24:02d}:{m:02d}"
            except: return np.nan
        try:
            m = round(float(x) * 24 * 60)
            return f"{(m//60)%24:02d}:{m%60:02d}"
        except: return np.nan

    def _time_of_day(self, time_str):
        """Buckets time into periods."""
        if pd.isna(time_str): return "unknown"
        try:
            t = pd.to_datetime(time_str, format="%H:%M", errors="coerce")
            if pd.isna(t): return "unknown"
            h = t.hour
            if 5 <= h < 8: return "early_morning"
            elif 8 <= h < 11: return "breakfast"
            elif 11 <= h < 14: return "lunch_peak"
            elif 14 <= h < 17: return "afternoon"
            elif 17 <= h < 20: return "evening_snacks"
            elif 20 <= h < 23: return "dinner_peak"
            else: return "late_night"
        except: return "unknown"



        