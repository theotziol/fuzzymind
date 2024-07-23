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


def weight_matrix_results(fold = None):
    if st.session_state.learning_algorithm == 'Neural-FCM':
        if fold == None:
            x_test = st.session_state.input_df.iloc[int(len(st.session_state.input_df)*st.session_state.split_ratio):]
            y_test = st.session_state.output_df.iloc[int(len(st.session_state.output_df)*st.session_state.split_ratio):]
            weight_matrix_widgets_neuralfcm(st.session_state.model, x_test, y_test)
        else:
            model = st.session_state.kfold_dic[fold]
            test_index = st.session_state.kfold_dic[fold].test_index
            x_test = st.session_state.input_df.iloc[test_index]
            y_test = st.session_state.output_df.iloc[test_index]

            weight_matrix_widgets_neuralfcm(model, x_test, y_test)
    elif st.session_state.learning_algorithm == 'Particle Swarm Optimization':
        pass



def weight_matrix_widgets_neuralfcm(model, x_test, y_test):
    st.info('Neural-FCM predicts a weight matrix for each instance.', icon = 'ℹ️')
    col1, col2 = st.columns(2)
    with col1:
        index = st.slider('Select an instance from the test dataset', 1, len(x_test), 1)
        st.caption(f'Input: Testing instance {index}')
        st.dataframe(x_test.iloc[index])
        ### to do change the st.write for regression-forecasting
        if st.session_state.learning_task == 'Classification':
            st.write(f'**Actual Class**: {y_test.columns[np.argmax(y_test.iloc[index])]}')
            st.write(f'**Predicted Class**: {y_test.columns[np.argmax(model.predictions[index])]}, {np.round(model.predictions[index], 2)}')
        else:
            st.write(f'**Actual value (norm)**: {np.round(model.test_y[index], 3)}')
            st.write(f'**Predicted value (norm)**: {np.round(model.predictions[index], 3)}, ')
            st.write(f'**Actual value**: {model.real_array_test[index]}')
            st.write(f'**Predicted value**: {np.round(model.predictions_actual[index], 3)}, ')
        
    with col2:
        fig_size = st.slider('Change figure size', 2, 10, 5)
        size = st.slider('Change figure text size',5, 30, 8)
        fig, axs = plt.subplots(figsize = (fig_size, fig_size))
        im, cbar = heatmap(model.predicted_matrices[index, :, :, 0], x_test.columns, x_test.columns, ax = axs, cmap = 'coolwarm')
        texts = annotate_heatmap(im, size)
        st.caption('FCM weight matrix.')
        st.pyplot(fig)

    graph_boolean = st.toggle('Generate FCM graph', help = 'Visualize as a graph the weight matrix.')
    if graph_boolean:
        graph(pd.DataFrame(np.round(model.predicted_matrices[index, :, :, 0], 2), index = x_test.columns, columns = x_test.columns))