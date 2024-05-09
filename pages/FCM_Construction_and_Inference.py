import streamlit as st

import sys
sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')

from fcm_codes.fcm_class import FCM_numpy
from app_components.inference_parameters import *
from app_components.fcm_graph_component import *
from app_components.inference_tab import *
from app_components.file_upload import *
from app_components.design_manually import *

# General Page Configurations
st.set_page_config(
    page_title = 'FCM Construction and Inference',
    page_icon="🛠",
    layout = 'wide',
    menu_items = {
        "Get Help" : None, #todo insert the github link
        "Report a Bug" : "mailto:tziolasphd@gmail.com", 
        "About" : "This app was created by Dr. Theodoros Tziolas under the supervision of Prof. Elpiniki Papageorgiou. It aims to provide a usefull AI tool that utilizes Fuzzy Cognitive Maps and Deep Learning for decision making."
        }
        )

st.sidebar.success("Select a tool above.")
st.title('FCM Construction and Inference 🛠')



tab_design, tab_inference = st.tabs(['FCM Design', 'Inference'])

matrix_exist = False
# Code for tab expert
with tab_design:

    st.subheader('Construct an FCM either by manually defining the concepts and the weighted interconnections, or automatically by uploading a weight matrix.', divider = 'blue')
    mode = st.radio('Select the designing mode', ['Design Manually', 'File Upload'], captions = ['Define concepts and interconnections manually', 'Upload a .csv that contains the weight matrix'], horizontal= True )

    if mode == 'Design Manually':
        edited_matrix, exists = manual_tab()
        if exists:
            graph_boolean  = st.radio('Generate FCM graph', ['No', 'Yes'], index = 0, horizontal = True)
            if graph_boolean == 'Yes':
                graph(edited_matrix)
            matrix_exist = True

    else: 
        edited_matrix, file = matrix_upload()
        if edited_matrix is not None:
            matrix_exist = True
            graph_boolean  = st.radio('Generate FCM graph', ['No', 'Yes'], index = 0, horizontal = True)
            if graph_boolean == 'Yes':
                graph(edited_matrix)

  

with tab_inference:

    if matrix_exist:
        
        num_iter, equilibrium, rule_text, activation_text, l , b = inference_parameters()
        initial_vector = define_initial_values(edited_matrix, activation_text)
        if np.all(initial_vector == 0):
            st.write('Pass an initial state vector')
        else:
            button = st.button('Inference')
            fcm = FCM_numpy(initial_vector, edited_matrix, num_iter, equilibrium, activation_text, rule_text, l, b)
            if button:
                inference = fcm.inference()
                # for i, n in enumerate(inference):
                #     print(i,np.round(n, 4))
                # print('\n')
                placeholder = inference_results(inference)
                button_clear = st.button('Clear')
                if button_clear:
                    placeholder.empty()

        

    else:
        non_matrix_definition()



    
    









