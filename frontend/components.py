import pandas as pd
import streamlit as st

try:
    from frontend.city_registry import CityRegistry
    from frontend.state import SessionStateManager
    from frontend.utils import TimeWindowUtility
except ImportError:
    from city_registry import CityRegistry
    from state import SessionStateManager
    from utils import TimeWindowUtility


class UIComponents:
    """Renders modular UI sections, forms, metrics, and cards."""

    @staticmethod
    def render_hero_banner():
        """Renders the top branding hero banner."""
        st.markdown("""
        <div class="hero-container">
            <div class="top-badge-row">
                <div class="brand-badge">
                    <span class="pulse-dot"></span> AI LOGISTICS ENGINE
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <div class="stat-pill">Stacking Ensemble (LightGBM + RF)</div>
                    <div class="stat-pill">Mean Error: <strong>±3.2 mins</strong></div>
                    <div class="stat-pill">Latency: <strong>&lt;15ms</strong></div>
                    <div class="stat-pill"><strong>22 Indian Metros</strong></div>
                </div>
            </div>
            <div class="hero-title">Food Delivery Time Prediction</div>
            <div class="hero-subtitle">
                Calibrated arrival estimation powered by stacked machine learning. 
                Estimates door-to-door transit time considering distance, road traffic, weather conditions, and fleet capacity.
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_city_selector(current_city: str) -> str:
        """Renders quick-city action buttons, metropolitan dropdown, and status banner."""
        st.markdown("##### 🌆 Step 1: Select Operational Delivery City")

        # Quick Select Action Buttons
        top_cities = CityRegistry.TOP_HUBS
        cols = st.columns(len(top_cities))
        for i, c_name in enumerate(top_cities):
            with cols[i]:
                is_selected = (current_city == c_name)
                st.button(
                    f"{'● ' if is_selected else ''}{c_name}",
                    key=f"quick_city_btn_{c_name}",
                    on_click=SessionStateManager.set_quick_city,
                    args=(c_name,),
                    type="primary" if is_selected else "secondary",
                    use_container_width=True
                )

        # Full Dropdown for all supported metropolitan hubs
        all_cities = CityRegistry.get_all_city_names()
        st.selectbox(
            "Or select from all 22 supported metropolitan hubs:",
            options=all_cities,
            key="city_select_box",
            on_change=SessionStateManager.sync_from_selectbox
        )

        selected_city = SessionStateManager.get_current_city()
        city_data = CityRegistry.get_city_data(selected_city)
        lat_min, lat_max = city_data["lat_range"]
        lng_min, lng_max = city_data["lng_range"]

        st.markdown(f"""
        <div class="city-info-banner">
            <span style="font-size: 1.25rem;">🧭</span>
            <div>
                <strong>Operational Hub: {selected_city}</strong> &nbsp;|&nbsp; 
                Fleet Hub ID: <code>{city_data["hub_code"]}</code> &nbsp;|&nbsp; 
                City Type: <b>{city_data["city_type"]}</b><br>
                • <b>Metropolitan Coordinate Bounds:</b> Latitude <code>[{lat_min:.4f} – {lat_max:.4f}]</code>, Longitude <code>[{lng_min:.4f} – {lng_max:.4f}]</code><br>
                <span style="color: #647a94; font-size: 0.82rem;">
                    Coordinates are pre-filled for {selected_city}. You can adjust them directly below.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        return selected_city

    @staticmethod
    def render_route_metrics_card(
        calc_dist: float,
        driving_est: float,
        is_rest_in_city: bool,
        is_drop_in_city: bool,
        rest_lat: float,
        rest_lng: float,
        cust_lat: float,
        cust_lng: float,
        city_name: str
    ):
        """Renders route indicators, perimeter status, and expander route map."""
        dist_col_a, dist_col_b = st.columns([2, 1])
        with dist_col_a:
            st.markdown(
                f"**Route Metrics:** <span class='route-dist-badge'>Straight-line: {calc_dist} km</span> "
                f"<span class='route-dist-badge'>Est. Road Path: ~{driving_est} km</span>",
                unsafe_allow_html=True
            )
        with dist_col_b:
            if is_rest_in_city and is_drop_in_city:
                st.markdown(
                    f"<span class='route-dist-badge' style='color:#7ad1a2; border-color:rgba(52,168,110,0.3); background:rgba(30,65,50,0.35);'>Within {city_name} Perimeter</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<span class='route-dist-badge' style='color:#d8b065; border-color:rgba(180,130,45,0.3); background:rgba(75,60,25,0.35);'>Outside Typical {city_name} Bounds</span>",
                    unsafe_allow_html=True
                )

        if calc_dist > 25.0:
            st.warning(f"Notice ({calc_dist} km): Route exceeds standard delivery radius (<20 km).")

        with st.expander("Preview Route Map (Restaurant & Customer Drop)", expanded=False):
            map_data = pd.DataFrame([
                {"lat": rest_lat, "lon": rest_lng, "label": "Restaurant"},
                {"lat": cust_lat, "lon": cust_lng, "label": "Customer Drop"}
            ])
            st.map(map_data, zoom=12, use_container_width=True)

    @staticmethod
    def render_prediction_results_showcase(
        eta_minutes: int,
        calc_dist: float,
        traffic: str,
        weather: str,
        rider_rating: float,
        time_ordered: str,
        city_name: str
    ):
        """Displays the calculated ETA card and context metric insight tiles."""
        tier_class, tier_label = TimeWindowUtility.classify_speed_tier(eta_minutes)
        arrival_window = TimeWindowUtility.compute_arrival_window(time_ordered, eta_minutes, city_name)

        # Luminous ETA Card
        st.markdown(f"""
        <div class="result-container">
            <span class="pill-badge {tier_class}">{tier_label}</span>
            <div class="eta-display">{eta_minutes}</div>
            <div class="eta-unit-label">MINUTES TOTAL TRANSIT</div>
            <div class="arrival-text">{arrival_window}</div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Context Metric Insight Tiles
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="insight-tile">
                <div class="insight-label">Route Distance</div>
                <div class="insight-value">{calc_dist} km</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="insight-tile">
                <div class="insight-label">Traffic State</div>
                <div class="insight-value">{traffic}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="insight-tile">
                <div class="insight-label">Weather State</div>
                <div class="insight-value">{weather}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="insight-tile">
                <div class="insight-label">Rider Rating</div>
                <div class="insight-value">★ {rider_rating}</div>
            </div>
            """, unsafe_allow_html=True)
