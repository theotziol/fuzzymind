import streamlit as st
from PIL import Image
from app_components.footer import *
from app_components.sidebar import *
import tensorflow as tf


# Page Configuration
st.set_page_config(
    page_title="FCM-App",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/theotziol/fcm-app/blob/master/Manual.docx",
        "Report a Bug": "mailto:ttziolasd@uth.gr",
        "About": "Developed by Dr. Theodoros Tziolas under Prof. Elpiniki Papageorgiou, this app provides AI-powered decision-making using Fuzzy Cognitive Maps (FCM) and Deep Learning.",
    },
)


# Custom CSS for Styling
st.markdown(
    """
    <style>
        .main-title { text-align: center; font-size: 40px; font-weight: bold; color: #4A90E2; }
        .subtext { text-align: center; font-size: 18px; color: #666; }
        .highlight { color: #E94E77; font-weight: bold; }
        .side-info { font-size: 16px; color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<p class="main-title">Welcome to FCM-App! 👋</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtext">An AI-powered tool for decision-making using <span class="highlight">Fuzzy Cognitive Maps</span> and Deep Learning.</p>',
    unsafe_allow_html=True,
)

# About Section
st.markdown(
    """
    ### 🔍 What is FCM-App?
    FCM-App is an open-source web application built with the [Streamlit](https://docs.streamlit.io) framework. It enables **Artificial Intelligence (AI)-driven decision-making** by utilizing:
    
    - 🧠 **Fuzzy Cognitive Maps (FCM) for modeling complex systems**
    - ⚡ **Neural-FCM algorithm** for optimizing weight matrices

    **👈 Use the sidebar** to explore the app’s features!

     Developed by **Dr. Theodoros Tziolas** under the guidance of [Prof. Elpiniki Papageorgiou](https://www.energy.uth.gr/index.php/en/personnel/papageorgiou-elpiniki.html), director of [ACTA](https://acta.energy.uth.gr/) laboratory, of University of Thessaly.
    """
)


# Contact & Contribution Section
st.markdown(
    """
    ### ❓ Have Questions or Found a Bug?
    - 📧 Contact via [email](mailto:ttziolas@uth.gr)
    - 🚀 Check out more projects on [GitHub](https://github.com/theotziol)
    - 🔬 Explore research from [ACTA Lab](https://acta.energy.uth.gr/)
    - 🤝 Contribute! We welcome new FCM learning algorithms such as **population-based** and **Hebbian-based methods**.
    """
)
# Centered Image
image = Image.open("imgs/AdobeStock_320559014.jpeg")
st.image(image, use_column_width=True)
# Sidebar Enhancement
sidebar_help_home()
sidebar_logo()

# Footer
st.markdown(footer_markdown(), unsafe_allow_html=True)
