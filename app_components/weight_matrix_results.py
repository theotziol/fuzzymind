### This script contains the widgets for the weight_matrix results tab


import streamlit as st 
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import sys

sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from fcm_codes.graphs import *
from app_components.fcm_graph_component import *



def weight_matrix_results():
    if st.session_state.learning_algorithm == 'Neural-FCM':
        if st.session_state.split_method == 'KFold':
            pass
        else:
            weight_matrix_widgets_neuralfcm()
    elif st.session_state.learning_algorithm == 'Particle Swarm Optimization':
        pass



def weight_matrix_widgets_neuralfcm():
    st.info('Neural-FCM predicts a weight matrix for each instance.', icon = 'ℹ️')
    col1, col2 = st.columns(2)
    x_test = st.session_state.input_df.iloc[int(len(st.session_state.input_df)*st.session_state.split_ratio):]
    y_test = st.session_state.output_df.iloc[int(len(st.session_state.output_df)*st.session_state.split_ratio):]
    with col1:
        index = st.slider('Select an instance from the test dataset', 1, len(st.session_state.output_df.iloc[int(len(st.session_state.output_df)*st.session_state.split_ratio):]), 1)
        st.caption(f'Input: Instance {index}')
        st.dataframe(x_test.iloc[index])
        ### to do change the st.write for regression-forecasting
        st.write(f'**Actual Class**: {y_test.columns[np.argmax(y_test.iloc[index])]}')
        st.write(f'**Predicted Class**: {y_test.columns[np.argmax(st.session_state.model.predictions[index])]}, {np.round(st.session_state.model.predictions[index], 2)}')
        
    with col2:
        fig_size = st.slider('Change figure size', 2, 10, 5)
        size = st.slider('Change figure text size',5, 30, 8)
        fig, axs = plt.subplots(figsize = (fig_size, fig_size))
        im, cbar = heatmap(st.session_state.model.predicted_matrices[index, :, :, 0], x_test.columns, x_test.columns, ax = axs, cmap = 'coolwarm')
        texts = annotate_heatmap(im, size)
        st.caption('FCM weight matrix.')
        st.pyplot(fig)

    graph_boolean = st.toggle('Generate FCM graph', help = 'Visualize as a graph the weight matrix.')
    if graph_boolean:
        graph(pd.DataFrame(np.round(st.session_state.model.predicted_matrices[index, :, :, 0], 2), index = x_test.columns, columns = x_test.columns))