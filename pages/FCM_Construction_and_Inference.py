import streamlit as st

import sys
sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from fcm_codes.general_functions import *
from fcm_codes.fcm_class import FCM_numpy
from app_components.inference_parameters import *
from app_components.fcm_graph_component import *
from app_components.inference_tab import *

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
    mode = st.radio('Select the designing mode', ['Design Manually', 'File Upload'], captions = ['Define concepts and interconnections manually', 'Upload a file that contains the weight matrix'], horizontal= True )

    if mode == 'Design Manually':
        #todo aggregation of fuzzy (csv)
        st.subheader('Define the total number of concepts', divider = 'green')
        num_concepts = st.number_input('Give the number of concepts', min_value=3, max_value=50, value = None, help = 'Give an integer in the range [3, 50]')
        if num_concepts != None:
            st.subheader('Define concepts', divider = 'green')
            columns_df = create_weight_matrix_columns(num_concepts)
            edited_columns = st.data_editor(columns_df, hide_index=True)
            st.subheader('Define weighted interconnections', divider = 'green')
            weight_matrix_df = create_weight_matrix(num_concepts, edited_columns.values.tolist())
            
            edited_matrix = st.data_editor(weight_matrix_df.style.apply(highlight_diagonal, axis=None), hide_index=True, disabled = ['-'], column_config=fix_configs(weight_matrix_df))
            edited_matrix.set_index('-', inplace = True)
            edited_matrix = edited_matrix.astype(float)
            graph_boolean  = st.radio('Generate FCM graph', ['No', 'Yes'], index = 0, horizontal = True)
            if graph_boolean == 'Yes':
                graph(edited_matrix)
            matrix_exist = True

    else: 
        weight_matrix = st.file_uploader('Upload a .csv or .xlsx')
        # edited_matrix = ...
        # matrix_exist = True
        #todo give preprocessing options/delimeter info etc
    

    

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



    
    









