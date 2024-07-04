import streamlit as st 
import pandas as pd 


help_learning_algorithms = "Neural-FCM is an FCM learning algorithm introduced by (Tziolas et al 2024) that utilizes a neural network for learning the FCM matrix.\
    Particle Swarm Optimization (PSO) is a population-based algorithm widely used in FCM learning tasks"


def learning_method_widgets():
    st.subheader('Learning methods', divider = 'blue')
    col1, col2, col3 = st.columns([0.3,0.4,0.3])
    with col2:
        st.radio('Select learning algorithm', ['Neural-FCM', 'Particle Swarm Optimization'], None, help = help_learning_algorithms, key = 'learning_algorithm', horizontal=True)
    
    tab_param, tab_results = st.tabs(['📓 Training parameters', '📋 Results']) 
    


    
        



def parameters_tab(task, algorithm):
    if task == 'Classification':
        pass

    else:
        pass