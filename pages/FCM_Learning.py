import streamlit as st 

import sys
sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from app_components.data_upload import *
from app_components.preprocessing_tab import *

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


if 'uploaded' not in st.session_state.keys():
    st.session_state.uploaded = False


data_tab, data_visual, preprocessing_tab, learning_tab = st.tabs(['📂 Data Upload', '📈 Data Visualization', '⚙️ Data Preprocessing', '🧠 Learning'])

with data_tab:
    csv = upload_widgets()
    if csv is not None:
        if st.session_state.uploaded:
            st.sidebar.success(f'The {csv.name} file has been succesfully imported')
            modify_dataset()
    
            
    else:
        st.session_state.uploaded = False
        if 'working_df' in st.session_state.keys():
            del st.session_state.working_df
        
        
        st.sidebar.info('Import data to continue', icon="ℹ️")

with data_visual:
    if st.session_state.uploaded:
        plot_widgets()
    else:
        st.markdown(
        """
        👆 Use the **Data** tab to upload and import a dataset for learning.
        # ⛔ This tab will be accesible after data importing. ⛔ 
        """
        )


with preprocessing_tab:
    if st.session_state.uploaded:
            
        tab_cleansing, tab_transf, tab_norm, tab_split = st.tabs(
            ['🧹️ Data Cleansing', '🔨 Data Transformation', '⚖️ Data normalization', '✂️ Data Split']
            ) 
        with tab_cleansing:
            datacleansing_widgets()
    else:
        st.markdown(
        """
        👆 Use the **Data** tab to upload and import a dataset for learning.
        # ⛔ This tab will be accesible after data importing. ⛔ 
        """
        )

        


        



