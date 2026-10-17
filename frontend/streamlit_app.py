import sys
from pathlib import Path
from datetime import datetime
import streamlit as st

# Add current frontend directory to sys.path so imports work seamlessly in any environment
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import PAGE_CONFIG, BACKEND_URL
from styles import DesignSystem
from city_registry import CityRegistry
from utils import GeoMathUtility
from api_client import DeliveryPredictionClient
from state import SessionStateManager
from components import UIComponents


st.set_page_config(**PAGE_CONFIG)


def main():
    """Main application lifecycle controller."""
    # 1. Initialize State & Theme
    SessionStateManager.initialize()
    DesignSystem.inject()

    # 2. Render Header Hero
    UIComponents.render_hero_banner()

    # 3. Render City Selector
    current_city_name = SessionStateManager.get_current_city()
    selected_city = UIComponents.render_city_selector(current_city_name)
    city_data = CityRegistry.get_city_data(selected_city)
    hub_code = city_data["hub_code"]
    city_type = city_data["city_type"]
    lat_min, lat_max = city_data["lat_range"]
    lng_min, lng_max = city_data["lng_range"]

    # 4. Form Composition
    with st.form("prediction_form"):
        # Section 2.1: Geolocation Coordinates
        st.markdown(f"""
        <div class="card-header-title">
            <span class="card-header-icon">📍</span> 2. Geolocation Coordinates Input ({selected_city})
        </div>
        """, unsafe_allow_html=True)

        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            rest_lat = st.number_input(
                "Restaurant Latitude",
                value=float(city_data["default_rest"][0]),
                min_value=6.0000,
                max_value=38.0000,
                step=0.0001,
                format="%.4f",
                key=f"rest_lat_{selected_city}",
                help=f"Latitude for pickup restaurant (Typical {selected_city}: {lat_min:.2f} to {lat_max:.2f})"
            )
        with r_col2:
            rest_lng = st.number_input(
                "Restaurant Longitude",
                value=float(city_data["default_rest"][1]),
                min_value=68.0000,
                max_value=98.0000,
                step=0.0001,
                format="%.4f",
                key=f"rest_lng_{selected_city}",
                help=f"Longitude for pickup restaurant (Typical {selected_city}: {lng_min:.2f} to {lng_max:.2f})"
            )
        with r_col3:
            cust_lat = st.number_input(
                "Customer Drop Latitude",
                value=float(city_data["default_drop"][0]),
                min_value=6.0000,
                max_value=38.0000,
                step=0.0001,
                format="%.4f",
                key=f"cust_lat_{selected_city}",
                help=f"Latitude for customer drop-off (Typical {selected_city}: {lat_min:.2f} to {lat_max:.2f})"
            )
        with r_col4:
            cust_lng = st.number_input(
                "Customer Drop Longitude",
                value=float(city_data["default_drop"][1]),
                min_value=68.0000,
                max_value=98.0000,
                step=0.0001,
                format="%.4f",
                key=f"cust_lng_{selected_city}",
                help=f"Longitude for customer drop-off (Typical {selected_city}: {lng_min:.2f} to {lng_max:.2f})"
            )

        calc_dist = GeoMathUtility.haversine_distance(rest_lat, rest_lng, cust_lat, cust_lng)
        driving_est = GeoMathUtility.estimate_road_distance(calc_dist)
        is_rest_in_city = CityRegistry.is_within_bounds(selected_city, rest_lat, rest_lng)
        is_drop_in_city = CityRegistry.is_within_bounds(selected_city, cust_lat, cust_lng)

        UIComponents.render_route_metrics_card(
            calc_dist, driving_est, is_rest_in_city, is_drop_in_city,
            rest_lat, rest_lng, cust_lat, cust_lng, selected_city
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Section 2.2: Fleet & Rider Profile
        st.markdown("""
        <div class="card-header-title">
            <span class="card-header-icon">🛵</span> 3. Delivery Fleet & Partner Profile
        </div>
        """, unsafe_allow_html=True)

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            rider_age = st.number_input("Rider Age", min_value=18, max_value=70, value=28)
        with p_col2:
            rider_rating = st.number_input("Rider Rating (1.0 – 5.0)", min_value=1.0, max_value=5.0, value=4.8, step=0.1)
        with p_col3:
            vehicle_type = st.selectbox(
                "Vehicle Type",
                options=["motorcycle", "scooter", "electric_scooter", "bicycle"],
                format_func=lambda v: {
                    "motorcycle": "Motorcycle",
                    "scooter": "Scooter",
                    "electric_scooter": "Electric Scooter",
                    "bicycle": "Bicycle"
                }.get(v, v),
                index=0
            )

        p_col4, p_col5 = st.columns(2)
        with p_col4:
            vehicle_cond = st.selectbox(
                "Vehicle Condition Rating",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 (Needs Service)",
                    1: "1 (Fair)",
                    2: "2 (Good / Standard)",
                    3: "3 (Pristine Condition)"
                }[x],
                index=2
            )
        with p_col5:
            multiple_del = st.selectbox(
                "Active Batch Deliveries",
                options=[0, 1, 2, 3],
                format_func=lambda x: f"{x} Parallel Drop (Single Dedicated)" if x == 0 else f"{x} Active Batched Deliveries",
                index=0
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Section 2.3: Environment & Traffic
        st.markdown("""
        <div class="card-header-title">
            <span class="card-header-icon">🌦️</span> 4. Environmental & Traffic Dynamics
        </div>
        """, unsafe_allow_html=True)

        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            weather = st.selectbox(
                "Weather Condition",
                options=["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"],
                format_func=lambda w: {
                    "Sunny": "Sunny / Clear",
                    "Cloudy": "Overcast / Cloudy",
                    "Windy": "High Wind",
                    "Fog": "Dense Fog",
                    "Stormy": "Thunderstorm / Rain",
                    "Sandstorms": "Dust / Sandstorm"
                }[w],
                index=0
            )
        with e_col2:
            traffic = st.selectbox(
                "Road Traffic Density",
                options=["Low", "Medium", "High", "Jam"],
                format_func=lambda t: {
                    "Low": "Low (Free Flowing)",
                    "Medium": "Medium (Moderate)",
                    "High": "High (Heavy Peak)",
                    "Jam": "Jam (Gridlock)"
                }[t],
                index=2
            )
        with e_col3:
            festival = st.selectbox(
                "Festival / Public Surge Period",
                options=["No", "Yes"],
                format_func=lambda f: "Yes (Festival Rush)" if f == "Yes" else "No (Standard Day)",
                index=0
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Section 2.4: Order Details
        st.markdown("""
        <div class="card-header-title">
            <span class="card-header-icon">📦</span> 5. Order Information & Dispatch Time
        </div>
        """, unsafe_allow_html=True)

        order_id = "ORD-92841"
        o_col1, o_col2, o_col3 = st.columns(3)
        with o_col1:
            order_date = st.text_input("Order Date (DD-MM-YYYY)", value=datetime.today().strftime("%d-%m-%Y"))
        with o_col2:
            time_ordered = st.text_input("Order Placed Time (HH:MM)", value=datetime.now().strftime("%H:%M"))
        with o_col3:
            order_type = st.selectbox(
                "Order Category",
                options=["Meal", "Snack", "Drinks", "Buffet"],
                format_func=lambda o: {
                    "Meal": "Full Meal",
                    "Snack": "Quick Snack",
                    "Drinks": "Beverages / Drinks",
                    "Buffet": "Buffet Feast"
                }[o],
                index=0
            )

        submit_btn = st.form_submit_button("Calculate Estimated Delivery Time", use_container_width=True)

    # 5. Prediction Execution & Presentation
    if submit_btn:
        payload = {
            "ID": order_id,
            "Delivery_person_ID": hub_code.strip(),
            "Delivery_person_Age": float(rider_age),
            "Delivery_person_Ratings": float(rider_rating),
            "Restaurant_latitude": float(rest_lat),
            "Restaurant_longitude": float(rest_lng),
            "Delivery_location_latitude": float(cust_lat),
            "Delivery_location_longitude": float(cust_lng),
            "Order_Date": order_date.strip(),
            "Time_Orderd": time_ordered.strip(),
            "Weather_conditions": str(weather),
            "Road_traffic_density": str(traffic),
            "Vehicle_condition": int(vehicle_cond),
            "Type_of_order": str(order_type),
            "Type_of_vehicle": str(vehicle_type),
            "multiple_deliveries": float(multiple_del),
            "Festival": str(festival),
            "City": str(city_type)
        }

        client = DeliveryPredictionClient(BACKEND_URL)
        with st.spinner("Running stacking ensemble inference..."):
            success, result, error_message = client.predict_delivery_time(payload)

            if success and result:
                eta_minutes = round(result["estimated_delivery_minutes"])
                UIComponents.render_prediction_results_showcase(
                    eta_minutes=eta_minutes,
                    calc_dist=calc_dist,
                    traffic=traffic,
                    weather=weather,
                    rider_rating=rider_rating,
                    time_ordered=time_ordered,
                    city_name=selected_city
                )
            else:
                st.error(error_message)


if __name__ == "__main__":
    main()
