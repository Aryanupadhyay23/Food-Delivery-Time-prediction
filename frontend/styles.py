"""
Custom CSS Design System & Ambient Midnight Blue Theme Tokens
"""

import streamlit as st


class DesignSystem:
    """Manages custom UI styling and dark theme tokens for the application."""

    @staticmethod
    def get_custom_css() -> str:
        return """
        <style>
            /* Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', sans-serif;
            }
            
            h1, h2, h3, h4, h5, h6, .outfit-font {
                font-family: 'Outfit', sans-serif !important;
                font-weight: 600 !important;
                color: #e2e8f0 !important;
            }

            /* Ambient Sapphire-Lit Canvas Backdrop */
            .stApp {
                background: radial-gradient(ellipse 90% 50% at 50% -5%, rgba(59, 130, 246, 0.18) 0%, transparent 65%),
                            radial-gradient(circle 600px at 85% 15%, rgba(37, 99, 235, 0.12) 0%, transparent 60%),
                            radial-gradient(circle 500px at 15% 45%, rgba(29, 78, 216, 0.08) 0%, transparent 50%),
                            #090d16 !important;
                color: #94a3b8;
            }

            /* Streamlit Global Text Override */
            div[data-testid="stMarkdownContainer"] p {
                color: #94a3b8;
                font-size: 0.94rem;
                line-height: 1.55;
            }

            /* Luminous Code Badges */
            code {
                color: #93c5fd !important;
                background: rgba(37, 99, 235, 0.15) !important;
                border: 1px solid rgba(96, 165, 250, 0.3) !important;
                padding: 2px 7px !important;
                border-radius: 5px !important;
                font-size: 0.86em !important;
            }

            /* Top Brand Hero Banner Card (Glassmorphic Sheen) */
            .hero-container {
                padding: 22px 26px;
                background: linear-gradient(180deg, rgba(16, 25, 46, 0.82) 0%, rgba(10, 16, 28, 0.9) 100%);
                border: 1px solid rgba(75, 120, 185, 0.28);
                border-top: 1px solid rgba(147, 197, 253, 0.4);
                border-radius: 16px;
                margin-bottom: 22px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35), 0 0 30px rgba(37, 99, 235, 0.08);
                backdrop-filter: blur(12px);
            }

            .top-badge-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 10px;
            }

            .brand-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(37, 99, 235, 0.16);
                color: #93c5fd;
                border: 1px solid rgba(96, 165, 250, 0.35);
                padding: 4px 13px;
                border-radius: 99px;
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.5px;
                box-shadow: 0 0 14px rgba(59, 130, 246, 0.15);
            }

            .pulse-dot {
                width: 7px;
                height: 7px;
                background-color: #34d399;
                border-radius: 50%;
                display: inline-block;
                box-shadow: 0 0 10px rgba(52, 211, 153, 0.7);
            }

            .stat-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(14, 22, 38, 0.8);
                border: 1px solid rgba(75, 105, 145, 0.28);
                padding: 4px 11px;
                border-radius: 8px;
                font-size: 0.78rem;
                color: #94a3b8;
            }

            .stat-pill strong {
                color: #e2e8f0;
            }

            .hero-title {
                font-family: 'Outfit', sans-serif;
                font-size: 2.35rem;
                font-weight: 700;
                background: linear-gradient(135deg, #ffffff 10%, #cbd5e1 50%, #93c5fd 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                line-height: 1.2;
                margin-bottom: 6px;
                letter-spacing: -0.3px;
            }

            .hero-subtitle {
                color: #94a3b8;
                font-size: 0.94rem;
                line-height: 1.55;
                max-width: 840px;
                margin-bottom: 2px;
            }

            /* Section Card Header Title */
            .card-header-title {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 1.08rem;
                font-weight: 600;
                color: #e2e8f0;
                margin-bottom: 12px;
            }

            .card-header-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 30px;
                height: 30px;
                border-radius: 8px;
                background: rgba(37, 99, 235, 0.18);
                border: 1px solid rgba(96, 165, 250, 0.35);
                color: #93c5fd;
                font-size: 1rem;
                box-shadow: 0 0 12px rgba(59, 130, 246, 0.18);
            }

            /* Glowing City Guidance Banner */
            .city-info-banner {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                background: linear-gradient(180deg, rgba(16, 26, 46, 0.75) 0%, rgba(10, 16, 30, 0.85) 100%);
                border: 1px solid rgba(75, 120, 185, 0.25);
                border-left: 3px solid #3b82f6;
                border-radius: 12px;
                padding: 14px 18px;
                margin: 10px 0 18px 0;
                font-size: 0.88rem;
                line-height: 1.5;
                color: #94a3b8;
                box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
            }

            .city-info-banner strong {
                color: #93c5fd;
            }

            /* Quick City Selector Buttons */
            div[data-testid="column"] button {
                background: rgba(16, 24, 42, 0.75) !important;
                border: 1px solid rgba(70, 100, 145, 0.3) !important;
                border-top: 1px solid rgba(147, 197, 253, 0.15) !important;
                color: #94a3b8 !important;
                border-radius: 9px !important;
                font-weight: 500 !important;
                font-size: 0.88rem !important;
                transition: all 0.2s ease !important;
            }

            div[data-testid="column"] button:hover {
                border-color: rgba(96, 165, 250, 0.5) !important;
                background: rgba(28, 44, 76, 0.7) !important;
                color: #e2e8f0 !important;
                box-shadow: 0 0 14px rgba(59, 130, 246, 0.2) !important;
            }

            /* Active Selected City Button */
            div[data-testid="column"] button[kind="primary"],
            div[data-testid="column"] button[data-testid*="primary"] {
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.4) 0%, rgba(30, 58, 138, 0.55) 100%) !important;
                border: 1px solid rgba(96, 165, 250, 0.55) !important;
                border-top: 1px solid rgba(191, 219, 254, 0.55) !important;
                color: #ffffff !important;
                font-weight: 600 !important;
                box-shadow: 0 0 16px rgba(59, 130, 246, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
            }

            /* Result ETA Card (Luminous Showcase) */
            .result-container {
                background: linear-gradient(180deg, rgba(17, 28, 52, 0.85) 0%, rgba(10, 16, 32, 0.95) 100%);
                border: 1px solid rgba(96, 165, 250, 0.35);
                border-top: 1px solid rgba(147, 197, 253, 0.4);
                border-radius: 18px;
                padding: 28px 24px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 35px rgba(37, 99, 235, 0.14);
                backdrop-filter: blur(12px);
                margin: 20px 0 24px 0;
            }

            .eta-display {
                font-family: 'Outfit', sans-serif;
                font-size: 4.4rem;
                font-weight: 800;
                background: linear-gradient(180deg, #ffffff 15%, #93c5fd 95%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                line-height: 1.05;
                margin: 6px 0 2px 0;
                letter-spacing: -1px;
                filter: drop-shadow(0 0 20px rgba(96, 165, 250, 0.25));
            }

            .eta-unit-label {
                font-size: 0.92rem;
                color: #93c5fd;
                font-weight: 700;
                letter-spacing: 2px;
            }

            .arrival-text {
                color: #94a3b8;
                font-size: 0.96rem;
                margin-top: 12px;
            }

            .arrival-text b {
                color: #e2e8f0;
                font-weight: 600;
            }

            /* Luminous Status Badges */
            .pill-badge {
                display: inline-block;
                padding: 5px 16px;
                border-radius: 99px;
                font-size: 0.8rem;
                font-weight: 600;
                letter-spacing: 0.5px;
            }

            .pill-express {
                background: rgba(16, 185, 129, 0.15);
                color: #6ee7b7;
                border: 1px solid rgba(52, 211, 153, 0.4);
                box-shadow: 0 0 10px rgba(16, 185, 129, 0.15);
            }

            .pill-standard {
                background: rgba(37, 99, 235, 0.15);
                color: #93c5fd;
                border: 1px solid rgba(96, 165, 250, 0.4);
                box-shadow: 0 0 10px rgba(59, 130, 246, 0.15);
            }

            .pill-delay {
                background: rgba(245, 158, 11, 0.15);
                color: #fcd34d;
                border: 1px solid rgba(251, 191, 36, 0.4);
                box-shadow: 0 0 10px rgba(245, 158, 11, 0.15);
            }

            .route-dist-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(37, 99, 235, 0.12);
                color: #93c5fd;
                border: 1px solid rgba(96, 165, 250, 0.3);
                padding: 4px 11px;
                border-radius: 7px;
                font-weight: 500;
                font-size: 0.86rem;
            }

            /* 4-Column Metric Insight Tiles */
            .insight-tile {
                background: linear-gradient(180deg, rgba(16, 26, 48, 0.7) 0%, rgba(10, 17, 32, 0.8) 100%);
                border: 1px solid rgba(70, 105, 150, 0.25);
                border-top: 1px solid rgba(147, 197, 253, 0.18);
                border-radius: 13px;
                padding: 16px 18px;
                text-align: center;
                margin-bottom: 12px;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
                transition: all 0.2s ease;
            }

            .insight-tile:hover {
                border-color: rgba(96, 165, 250, 0.45);
                box-shadow: 0 6px 20px rgba(37, 99, 235, 0.15);
                transform: translateY(-1px);
            }

            .insight-label {
                font-size: 0.74rem;
                color: #798ea6;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.6px;
                margin-bottom: 5px;
            }

            .insight-value {
                font-family: 'Outfit', sans-serif;
                font-size: 1.35rem;
                font-weight: 700;
                color: #e2e8f0;
            }

            /* Primary Action Button */
            div[data-testid="stFormSubmitButton"] > button:first-child {
                background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #1e40af 100%) !important;
                border: 1px solid rgba(147, 197, 253, 0.35) !important;
                border-top: 1px solid rgba(255, 255, 255, 0.35) !important;
                color: #ffffff !important;
                font-family: 'Outfit', sans-serif !important;
                font-weight: 600 !important;
                font-size: 1.05rem !important;
                letter-spacing: 0.3px !important;
                border-radius: 12px !important;
                padding: 13px 26px !important;
                box-shadow: 0 4px 18px rgba(37, 99, 235, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
                transition: all 0.2s ease !important;
                margin-top: 10px !important;
            }

            div[data-testid="stFormSubmitButton"] > button:first-child:hover {
                background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #1d4ed8 100%) !important;
                border-color: rgba(191, 219, 254, 0.6) !important;
                color: #ffffff !important;
                box-shadow: 0 6px 24px rgba(59, 130, 246, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
                transform: translateY(-1px) !important;
            }

            /* Input Field Polishing */
            div[data-testid="stTextInput"] input, 
            div[data-testid="stNumberInput"] input, 
            div[data-testid="stSelectbox"] > div {
                border-radius: 9px !important;
                background-color: rgba(12, 19, 34, 0.85) !important;
                border: 1px solid rgba(70, 105, 150, 0.3) !important;
                color: #e2e8f0 !important;
                font-size: 0.93rem !important;
            }

            div[data-testid="stTextInput"] input:focus, 
            div[data-testid="stNumberInput"] input:focus, 
            div[data-testid="stSelectbox"] > div:focus-within {
                border-color: rgba(96, 165, 250, 0.65) !important;
                box-shadow: 0 0 12px rgba(59, 130, 246, 0.25) !important;
                color: #ffffff !important;
            }

            /* Widget Labels */
            label[data-testid="stWidgetLabel"] p,
            label[data-testid="stWidgetLabel"] span {
                color: #94a3b8 !important;
                font-size: 0.87rem !important;
                font-weight: 500 !important;
            }

            /* Clean Divider */
            hr {
                border: 0 !important;
                border-top: 1px solid rgba(70, 105, 150, 0.25) !important;
                margin: 16px 0 !important;
            }
        </style>
        """

    @classmethod
    def inject(cls) -> None:
        """Renders the CSS stylesheet into the Streamlit session."""
        st.markdown(cls.get_custom_css(), unsafe_allow_html=True)
