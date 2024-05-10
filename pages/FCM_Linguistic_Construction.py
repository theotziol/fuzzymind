import streamlit as st

import sys
sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from fcm_codes.general_functions import *
from fcm_codes.fcm_class import FCM_numpy
from app_components.inference_parameters import *
from app_components.fcm_graph_component import *
from app_components.inference_tab import *
from app_components.file_upload import *
from app_components.design_manually import *

# General Page Configurations
st.set_page_config(
    page_title = 'FCM Linguistic Construction and Inference',
    page_icon="🛠",
    layout = 'wide',
    menu_items = {
        "Get Help" : None, #todo insert the github link
        "Report a Bug" : "mailto:tziolasphd@gmail.com", 
        "About" : "This app was created by Dr. Theodoros Tziolas under the supervision of Prof. Elpiniki Papageorgiou. It aims to provide a usefull AI tool that utilizes Fuzzy Cognitive Maps and Deep Learning for decision making."
        }
        )

st.sidebar.success("Select a tool above.")
st.title('FCM Linguistic Construction and Inference')
st.header('Construct an FCM based on iinguistic variables and fuzzy sets theory.', divider = 'blue')

tab_design, tab_inference = st.tabs(['FCM Design', 'Inference'])

matrix_exist = False

# Code for tab expert
with tab_design:
    st.subheader('Construct the FCM and the fuzzy sets manually, upload a file, or upload multiple files for knowledge aggregation', divider = 'blue')
    mode = st.radio('Select the designing mode', ['Design Manually', 'File Upload', 'Knowledge Aggregation'], captions = ['Define concepts and interconnections manually', 'Upload a .csv that contains the weight matrix', 'Upload multiplr .csv files for knowledge aggregation'], horizontal= True )
    if mode == 'Design Manually':
        fuzzy_sets()
