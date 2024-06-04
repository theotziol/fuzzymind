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



tab1, tab2, tab3 = st.tabs(['Data', 'Preprocessing', 'Learning'])

if 'uploaded' not in st.session_state.keys():
    st.session_state.uploaded = False
    

with tab1:
    csv = upload_widgets()
    ### Currently there is an issue with saving the dictionairy
    if csv is not None:
        if st.session_state.uploaded == True:
            st.sidebar.success(f'The {csv.name} file has been succesfully imported')
            modify_dataset()
        else:
            st.sidebar.info('Import data to continue', icon="ℹ️")


    else:
        if st.session_state.uploaded == True:
            st.session_state.uploaded == False
        else:
            st.sidebar.info('Import data to continue', icon="ℹ️")
        


        



