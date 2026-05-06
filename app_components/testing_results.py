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
            testing_results_averaged_regression()
        elif fold == None:
            testing_results_regression(st.session_state.model)
        else:
            model = st.session_state.kfold_dic[fold]
            testing_results_regression(model)

    


def testing_results_classification(model, testing_samples):
    '''
    Function to plot the testing results of the classification:
    ---Currently has been tested for the Neural-FCM classifier---
    '''
    # 1. Polished KPI Metric Cards Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{model.accuracy:.4f}")
    m2.metric("F1 (Macro)", f"{model.f1_score_macro:.4f}")
    m3.metric("F1 (Micro)", f"{model.f1_score_micro:.4f}")
    m4.metric("Precision", f"{getattr(model, 'precision', 0):.4f}")
    m5.metric("Recall", f"{getattr(model, 'recall', 0):.4f}")
    m6.metric("AUC", f"{getattr(model, 'roc_auc', 0):.4f}" if getattr(model, 'roc_auc', None) is not None else "N/A")

    st.divider()

    # 2. Charts and Inference Stats Row
    col1, col2, col3 = st.columns([0.4, 0.4, 0.2])

    with col1:
        st.write("**Confusion Matrix**")
        disp = ConfusionMatrixDisplay(confusion_matrix=model.confusion_matrix, display_labels=st.session_state.output_columns)
        fig, axs = plt.subplots(figsize = (4, 4))
        disp.plot(cmap = 'hot', colorbar=False, ax = axs)
        st.pyplot(fig)

    with col2:
        st.write("**ROC Curve**")
        if hasattr(model, 'fpr') and model.fpr is not None:
            fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
            n_classes = len(model.fpr)
            for i in range(n_classes):
                # Map back to column names if available
                class_name = st.session_state.output_columns[i] if i < len(st.session_state.output_columns) else f"Class {i}"
                ax_roc.plot(model.fpr[i], model.tpr[i], label=f'{class_name} (AUC = {model.roc_auc_dict[i]:.2f})')
            
            ax_roc.plot([0, 1], [0, 1], 'k--', lw=1) # Diagonal line
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right", fontsize='x-small')
            st.pyplot(fig_roc)
        else:
            st.info("ROC Curve not available for this dataset.")

    with col3:
        st.write("**Inference Stats**")
        st.write(f'Total testing samples: {testing_samples}')
        b_size = np.min([testing_samples, 32]) #32 the default by keras model.predict
        st.write(f'Prediction time: {model.prediction_time} ms') 
        st.write(f'Batch size: {b_size}')


def testing_results_averaged():
    '''
    Aggregates results for K-Fold Cross Validation in a polished way.
    '''
    accuracy, f1_score_macro, f1_score_micro = [], [], []
    precision, recall, roc_auc, prediction_times = [], [], [], []
    
    for key in st.session_state.kfold_dic.keys():
        model = st.session_state.kfold_dic[key]
        accuracy.append(model.accuracy)
        f1_score_macro.append(model.f1_score_macro)
        f1_score_micro.append(model.f1_score_micro)
        precision.append(getattr(model, 'precision', np.nan))
        recall.append(getattr(model, 'recall', np.nan))
        roc_auc.append(getattr(model, 'roc_auc', np.nan))
        prediction_times.append(model.prediction_time)

    dic = {
        'Accuracy' : accuracy,
        'F1 (macro)' : f1_score_macro,
        'F1 (micro)' : f1_score_micro,
        'Precision' : precision,
        'Recall' : recall,
        'ROC AUC' : roc_auc,
        'Pred. times (ms)' : prediction_times
    }
    df = pd.DataFrame(dic, index = st.session_state.kfold_dic.keys())

    # Polished Average KPIs
    st.write("##### K-Fold Average Performance")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Avg Accuracy", f"{np.nanmean(accuracy):.4f}")
    m2.metric("Avg F1 (Macro)", f"{np.nanmean(f1_score_macro):.4f}")
    m3.metric("Avg F1 (Micro)", f"{np.nanmean(f1_score_micro):.4f}")
    m4.metric("Avg Precision", f"{np.nanmean(precision):.4f}")
    m5.metric("Avg Recall", f"{np.nanmean(recall):.4f}")
    m6.metric("Avg AUC", f"{np.nanmean(roc_auc):.4f}")

    st.divider()
    st.write("##### Detailed Fold Performance")
    st.dataframe(df.style.format("{:.4f}"))

def testing_results_regression(model):
    '''
    Function to plot the testing results of the classification:
    ---Currently has been tested for the Neural-FCM classifier---
    '''
    col1, col2, col3 = st.columns(3, gap = 'medium')

    with col1:
        st.write(f'**MSE**: {np.round(model.mse, 4)}\n')
        st.write(f'**MAE**: {np.round(model.mae, 4)}\n')
        st.write(f'**MAPE**: {np.round(model.mape, 2)}%\n')
        st.write(f'**$R^2$**: {np.round(model.R_sq, 4)}\n')

    with col2:
        st.write(f'**MSE (norm)**: {np.round(model.mse_norm, 4)}\n')
        st.write(f'**MAE (norm)**: {np.round(model.mae_norm, 4)}\n')
    with col3:
        st.write(f'**Total testing samples**: {len(model.real_array_test)}\n')
        b_size = np.min([len(model.real_array_test), 32]) #32 the default by keras model.predict
        st.write(f'**Total prediction time**: {model.prediction_time} ms\n') 
        st.write(f'**Prediction batch size**: {b_size}\n')
    
    tab_regress, tab_r  = st.tabs(['Fitting graph', '$R^2$ graph'])
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





def testing_results_averaged_regression():
    mse = []
    mae = []
    mape = []
    mse_norm = []
    mae_norm = []
    r_sq = []
    prediction_times = []
    for key in st.session_state.kfold_dic.keys():
        mse.append(st.session_state.kfold_dic[key].mse)
        mae.append(st.session_state.kfold_dic[key].mae)
        mape.append(st.session_state.kfold_dic[key].mape)
        mse_norm.append(st.session_state.kfold_dic[key].mse_norm)
        mae_norm.append(st.session_state.kfold_dic[key].mae_norm)
        r_sq.append(st.session_state.kfold_dic[key].R_sq)
        prediction_times.append(st.session_state.kfold_dic[key].prediction_time)

    dic = {
        'MSE' : mse,
        'MSE (norm)' : mse_norm,
        'MAE' : mae,
        'MAE (norm)' : mae_norm,
        'MAPE' : mape,
        'R^2' : r_sq,
        'Prediction times (ms)' : prediction_times
    }
    df = pd.DataFrame(dic, index = st.session_state.kfold_dic.keys())

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'**Average MSE**: {np.round(np.mean(mse), 4)}\n')
        st.write(f'**Average MAE**: {np.round(np.mean(mae), 4)}\n')
        st.write(f'**Average MSE (norm)**: {np.round(np.mean(mse_norm), 4)}\n')
        st.write(f'**Average MAE (norm)**: {np.round(np.mean(mae_norm), 4)}\n')
        st.write(f'**Average MAPE**: {np.round(np.mean(mape), 2)}%\n')
        st.write(f'**Average $R^2$**: {np.round(np.mean(r_sq), 4)}\n')
    with col2:
        st.dataframe(df)
