import streamlit as st 

import sys
sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from app_components.data_upload import *

st.set_page_config(
    page_title = 'FCM Learning',
    page_icon="🎓",
    layout = 'wide',
    menu_items = {
        "Get Help" : None, #todo insert the github link
        "Report a Bug" : "mailto:tziolasphd@gmail.com", 
        "About" : "This app was created by Dr. Theodoros Tziolas under the supervision of Prof. Elpiniki Papageorgiou. It aims to provide a usefull AI tool that utilizes Fuzzy Cognitive Maps and Deep Learning for decision making."
        }
        )

help_task = "Use the classification option if your FCM will categorize your data into distinct classes. \
    \nUse the regression option if your FCM will try to predict a continuous variable."

st.title('FCM Learning 🎓')
st.header('Construct an FCM based on data', divider = 'blue')


option = st.sidebar.radio('Select task', ['Classification', 'Regression'], help=help_task)
st.sidebar.success(f' You selected the {option} learning')

tab1, tab2, tab3 = st.tabs(['Data', 'Preprocessing', 'Learning'])

track_changes_limit = 500
track_changes_initial = 0

if 'track_changes_df' not in st.session_state:
    st.session_state.track_changes_df = track_changes_initial



with tab1:
    csv, df = upload_widgets()
    ### Currently there is an issue with saving the dictionairy
    if csv is not None:
        #use state to save dictionairies
        if 'initial_df' not in st.session_state:
            st.session_state.initial_df = df.copy()
        
        if 'processed_df' not in st.session_state:
            st.session_state.processed_df = df.copy()
    
        st.session_state.processed_df = modify_dataset(df)
        df = st.session_state.processed_df.copy()


    else:
        if 'initial_df' in st.session_state:
            del st.session_state['initial_df']
        
        if 'processed_df' in st.session_state:
            del st.session_state.processed_df


        



