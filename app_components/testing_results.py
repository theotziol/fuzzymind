### This script contains the widgets for the testing results tab

import streamlit as st 
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay
import sys

sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from fcm_codes.graphs import *


def testing_results():
    '''
    the function that invokes all the other functions when the learning is finished. 
    It aims to gather and show results from the testing dataset. 
    '''

    if st.session_state.learning_task == 'Classification':
        if st.session_state.split_method == 'KFold':
            pass
        else:
            st.caption('Testing results.')
            testing_results_standard_classification()
    elif st.session_state.learning_task == 'Regression':
        pass

    elif st.session_state.learning_task == 'Timeseries forecasting':
        pass


def testing_results_standard_classification():
    '''
    Function to plot the testing results of the classification:
    ---Currently has been tested for the Neural-FCM classifier---
    '''
    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        st.write(f'**Accuracy**: {np.round(st.session_state.model.accuracy, 4)}\n')
        st.write(f'**F1-score (macro)**: {np.round(st.session_state.model.f1_score_macro, 4)}\n')
        st.write(f'**F1-score (micro)**: {np.round(st.session_state.model.f1_score_micro, 4)}\n')

    with col2:
        disp = ConfusionMatrixDisplay(confusion_matrix=st.session_state.model.confusion_matrix, display_labels=st.session_state.output_columns)
        fig, axs = plt.subplots(figsize = (4, 4))
        disp.plot(cmap = 'hot', colorbar=False, ax = axs)
        st.pyplot(fig)

    with col3:
        st.write(f'Total testing samples: {len(st.session_state.output_df.iloc[int(len(st.session_state.output_df)*st.session_state.split_ratio):])}\n')
        b_size = np.min([len(st.session_state.output_df.iloc[int(len(st.session_state.output_df)*st.session_state.split_ratio):]), 32]) #32 the default by keras model.predict
        st.write(f'Total prediction time: {st.session_state.model.prediction_time} ms\n') 
        st.write(f'Prediction batch size: {b_size}\n')