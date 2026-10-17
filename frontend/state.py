"""
Session State Management & Reactive City Synchronizer
"""

import streamlit as st

try:
    from frontend.config import DEFAULT_CITY
except ImportError:
    from config import DEFAULT_CITY


class SessionStateManager:
    """Coordinates reactive state across city selectors and widgets."""

    @staticmethod
    def initialize():
        if "selected_city" not in st.session_state:
            st.session_state.selected_city = DEFAULT_CITY
        if "city_select_box" not in st.session_state:
            st.session_state.city_select_box = DEFAULT_CITY

    @staticmethod
    def set_quick_city(city_name: str):
        st.session_state.selected_city = city_name
        st.session_state.city_select_box = city_name

    @staticmethod
    def sync_from_selectbox():
        st.session_state.selected_city = st.session_state.city_select_box

    @staticmethod
    def get_current_city() -> str:
        return st.session_state.get("selected_city", DEFAULT_CITY)
