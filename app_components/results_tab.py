import streamlit as st 
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 


def results_widgets():
    '''
    The results will be plotted here
    '''
    st.subheader('Results', divider = 'blue')
    if st.session_state.training_finished:
        if st.session_state.split_method == 'KFold':
            tab_names = [f'Fold {i+1}' for i in range(st.session_state.kfold_n_splits)]
            tab_names.insert(0, 'Average Metrics')
            tabs = st.tabs(tab_names)
            for i, tab in enumerate(tabs):
                with tab:
                    pass 
    else:
        st.write('Learning results will be shown here after FCM learning finishes')