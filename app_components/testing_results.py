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


def testing_results(fold = None):
    '''
    the function that invokes all the other functions when the learning is finished. 
    It aims to gather and show results from the testing dataset. 
    '''
    st.caption('Testing results.')
    if st.session_state.learning_task == 'Classification':
        if fold == 'average':
            testing_results_averaged()
        elif fold == None:
            testing_samples = len(st.session_state.output_df.iloc[int(len(st.session_state.output_df)*st.session_state.split_ratio):])
            testing_results_classification(st.session_state.model, testing_samples)
        else:
            model = st.session_state.kfold_dic[fold]
            testing_samples = len(model.test_index)
            testing_results_classification(model, testing_samples)


    elif st.session_state.learning_task == 'Regression':
        if fold == 'average':
            testing_results_averaged()
        elif fold == None:
            testing_results_regression(st.session_state.model)
        else:
            model = st.session_state.kfold_dic[fold]
            testing_samples = len(model.test_index)
            testing_results_classification(model, testing_samples)

    


def testing_results_classification(model, testing_samples):
    '''
    Function to plot the testing results of the classification:
    ---Currently has been tested for the Neural-FCM classifier---
    '''
    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        st.write(f'**Accuracy**: {np.round(model.accuracy, 4)}\n')
        st.write(f'**F1-score (macro)**: {np.round(model.f1_score_macro, 4)}\n')
        st.write(f'**F1-score (micro)**: {np.round(model.f1_score_micro, 4)}\n')

    with col2:
        disp = ConfusionMatrixDisplay(confusion_matrix=model.confusion_matrix, display_labels=st.session_state.output_columns)
        fig, axs = plt.subplots(figsize = (4, 4))
        disp.plot(cmap = 'hot', colorbar=False, ax = axs)
        st.pyplot(fig)

    with col3:
        st.write(f'Total testing samples: {testing_samples}\n')
        b_size = np.min([testing_samples, 32]) #32 the default by keras model.predict
        st.write(f'Total prediction time: {model.prediction_time} ms\n') 
        st.write(f'Prediction batch size: {b_size}\n')

def testing_results_regression(model):
    '''
    Function to plot the testing results of the classification:
    ---Currently has been tested for the Neural-FCM classifier---
    '''
    col1, col2 = st.columns(2, gap = 'medium')

    with col1:
        st.write(f'**MSE**: {np.round(model.mse, 4)}\n')
        st.write(f'**MAE**: {np.round(model.mae, 4)}\n')
        st.write(f'**MAPE**: {np.round(model.mape, 4)}\n')
        st.write(f'**$R^2$**: {np.round(model.R_sq, 4)}\n')
        


    with col2:
        st.write(f'**MSE (norm)**: {np.round(model.mse_norm, 4)}\n')
        st.write(f'**MAE (norm)**: {np.round(model.mae_norm, 4)}\n')
        st.write(f'**Total testing samples**: {len(model.real_array_test)}\n')
        b_size = np.min([len(model.real_array_test), 32]) #32 the default by keras model.predict
        st.write(f'**Total prediction time**: {model.prediction_time} ms\n') 
        st.write(f'**Prediction batch size**: {b_size}\n')
    
    tab_regress, tab_r  = st.tabs(['Regression graph', '$R^2$ graph'])
    height = st.slider("Select figure's height", 3, 10, 4)
    width = st.slider("Select figure's width", 3, 10, 6)
    with tab_regress:
        # st.caption('Regression.')
        figs, axs = plt.subplots(figsize = (width, height ))
        axs.plot(model.real_array_test, label = 'Actual values')
        axs.plot(model.predictions_actual, label = 'Predicted values')

        axs.set_xlabel('Time')
        axs.legend()
        st.pyplot(figs)

    with tab_r:
        # st.caption(f'**$R^2$**')
        fig, ax = plt.subplots(figsize = (width, height))
        ax.scatter(model.real_array_test, model.predictions_actual)
        ax.plot(model.real_array_test, model.m * model.real_array_test + model.b, color = 'g')
        ax.set_xlabel('Actual values')
        ax.set_ylabel('Predicted values')
        st.pyplot(fig)

    
        

def testing_results_averaged():
    accuracy = []
    f1_score_macro = []
    f1_score_micro = []
    prediction_times = []
    for key in st.session_state.kfold_dic.keys():
        accuracy.append(st.session_state.kfold_dic[key].accuracy)
        f1_score_macro.append(st.session_state.kfold_dic[key].f1_score_macro)
        f1_score_micro.append(st.session_state.kfold_dic[key].f1_score_micro)
        prediction_times.append(st.session_state.kfold_dic[key].prediction_time)

    dic = {
        'Accuracy' : accuracy,
        'F1-Score (macro)' : f1_score_macro,
        'F1-Score (micro)' : f1_score_micro,
        'Prediction times (ms)' : prediction_times
    }
    df = pd.DataFrame(dic, index = st.session_state.kfold_dic.keys())

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'**Average accuracy**: {np.round(np.mean(accuracy), 4)}\n')
        st.write(f'**Average F1-score (macro)**: {np.round(np.mean(f1_score_macro), 4)}\n')
        st.write(f'**F1-score (micro)**: {np.round(np.mean(f1_score_micro), 4)}\n')
    with col2:
        st.dataframe(df)
