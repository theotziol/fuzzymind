import streamlit as st
from app_components.footer import *
from PIL import Image
import numpy as np 


st.set_page_config(
    page_title = 'FCM-app',
    page_icon="🧠",
    layout = 'wide',
    menu_items = {
        "Get Help" : None, #todo insert the github link
        "Report a Bug" : "mailto:tziolasphd@gmail.com", 
        "About" : "This app was created by Dr. Theodoros Tziolas under the supervision of Prof. Elpiniki Papageorgiou. It aims to provide a usefull AI tool that utilizes Fuzzy Cognitive Maps and Deep Learning for decision making."
        }
        )



st.write("# FCM-app! 👋")

# st.sidebar.success("Select a tool aboove.")

st.markdown(
    """
    FCM-app is an open-source web application built with [Streamlit](https://docs.streamlit.io) framework for Artificial Intelligence (AI) based decision making with 
    **Fuzzy Cognitive Maps (FCM) and/or Deep Learning**. It currently provides FCM construction and inference (both linguistic and numeric) and FCM learning with the novel Neural-FCM algorithm.
    
    This app was developed by Dr. Theodoros Tziolas during his Ph.D. and under the supervision of [Prof. Elpiniki Papageorgiou](https://www.energy.uth.gr/index.php/en/personnel/papageorgiou-elpiniki.html).  
    **👈 Navigate though the sidebar** to explore the app's capabilities! 
    
    **The development of this app was supported by ELIDEK and [EMERALD](https://emerald.uth.gr/) project!**
    ### Do you have any question or face a bug?
    - Contact via [email](emailto:ttziolas@uth.gr)
    - Jump into Theodoros Tziolas [github](https://github.com/theotziol) for more projects
    - Explore other research results  from [EMERALD](https://emerald.uth.gr/)!

"""
)
st.image(Image.open('imgs/AdobeStock_320559014.jpeg'))
st.markdown(footer_markdown(),unsafe_allow_html=True)

