import streamlit as st 
import io

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

### This session state variable indicates that a dataset has been uploaded
if 'uploaded' not in st.session_state.keys():
    st.session_state.uploaded = False

### This session state variable indicates that a dataset has preprocessed and is ready for learning
if 'initialized_preprocessing' not in st.session_state.keys():
    st.session_state.initialized_preprocessing = False

if 'normalized' not in st.session_state.keys():
    st.session_state.normalized = False

### This session state variable indicates that a training is completed
if 'training_finished' not in st.session_state.keys():
    st.session_state.training_finished = False

if 'input_df' not in st.session_state.keys():
    st.session_state.input_df = None

if 'output_df' not in st.session_state.keys():
    st.session_state.output_df = None

### This session state variable indicates that the training shall start. It is being toggled in the learning tab
if 'train' not in st.session_state.keys():
    st.session_state.train = False

### This is the session state variable where the trained model will be stored
if 'model' not in st.session_state.keys():
    st.session_state.model = None

  

### importing app components
import sys
sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from app_components.data_upload import *
from app_components.data_cleansing_tab import *
from app_components.visualization_tab import *
from app_components.data_transformation_tab import *
from app_components.data_norm_tab import *
from app_components.data_split_tab import *
from app_components.learning_tab import *
from app_components.sidebar import *


st.title('FCM Learning 🎓')
st.header('Construct an FCM based on data', divider = 'blue')


data_tab, data_visual, preprocessing_tab, learning_tab = st.tabs(['📂 Data Upload', '📈 Data Visualization', '⚙️ Data Preprocessing', '🧠 Learning'])



with data_tab:
    csv = upload_widgets()
    if csv is not None:
        if st.session_state.uploaded:
            st.sidebar.success(f'The {csv.name} file has been succesfully imported')
            modify_dataset()
    
            
    else:
        ### give back the initial values to the session state variables 
        # st.session_state.clear()
        st.session_state.uploaded = False
        st.session_state.initialized_preprocessing = False
        st.session_state.normalized = False
        st.session_state.training_finished = False
        st.session_state.output_df = None
        st.session_state.input_df = None
        st.session_state.model = None
        st.session_state.train = False

        if 'working_df' in st.session_state.keys():
            del st.session_state.working_df
        
        if 'non_norm_working_df' in st.session_state.keys():
            del st.session_state.non_norm_working_df
        
        

sidebar_widgets_task()

with data_visual:
    if st.session_state.uploaded:
        plot_widgets()
    else:
        st.markdown(
        """
        👆 Use the **📂 Data Upload** tab to upload and import a dataset for learning.
        # ⛔ This tab will be accesible after data importing. 
        """
        )


with preprocessing_tab:
    if st.session_state.uploaded:

        st.info('Select a processing step from the tabs below.')    
        tab_cleansing, tab_transf, tab_norm, tab_split = st.tabs(
            ['🧹️ Data Cleansing', '🔨 Data Transformation', '⚖️ Data Normalization', '✂️ Data Split']
            ) 
        with tab_cleansing:
            datacleansing_widgets()
        
        with tab_transf:
            transformation_widgets()

        with tab_norm:
            data_normalization()

        with tab_split:
            spliting_widgets()
        
        if st.session_state.changed:
            c_1, c_2, c_3 = st.columns(3)
            with c_3:
                st.write('')
                restore = st.button('Restore all changes', key = 'restored_changes', on_click=restore_df_changes_callback, help = 'This button will discard all the applied preprocessing methods, returning back the raw imported data')


    else:
        st.markdown(
        """
        👆 Use the **📂 Data Upload** tab to upload and import a dataset for learning.
        # ⛔ This tab will be accesible after data importing. 
        """
        )


with learning_tab:
    if st.session_state.initialized_preprocessing:
        learning_method_widgets()
        results_widgets()
    else:
        if not st.session_state.uploaded:
            st.markdown(
            """
            👆 Use the **📂 Data Upload** tab to upload and import a dataset for learning.
            # ⛔ This tab will be accesible after data importing. 
            """
            )
        else:
            st.markdown(
            """
            👆 Use the **⚙️ Data Preprocessing** tab to process data and to split input/output columns.
            # ⛔ This tab will be accesible after data splitting. 
            """
            )

sidebar_widgets_show_df()






