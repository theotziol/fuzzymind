### This script contains the widgets for the learning results tab

import streamlit as st 
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay
import sys

sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from fcm_codes.graphs import *
from app_components.fcm_graph_component import *



# def learning_results(fold = None):
#     '''
#     the function that invokes all the other functions when the learning is finished. 
#     It aims to gather and show results from learning. 
#     '''
    
    
#     if st.session_state.learning_algorithm == 'Neural-FCM':
#         if st.session_state.split_method == 'KFold':
#             pass
#         else:
#             st.caption('Learning results.')
#             learning_results_neuralfcm_standard(st.session_state.model, total_training_samples)
#     elif st.session_state.learning_algorithm == 'Particle Swarm Optimization':
#         pass

def learning_results(fold = None):
    '''
    the function that invokes all the other functions when the learning is finished. 
    It aims to gather and show results from learning. 
    '''
    st.caption('Learning results.')
    if st.session_state.learning_algorithm == 'Neural-FCM':
        if fold == 'average':
            learning_results_averaged()
        elif fold == None:
            total_training_samples = len(st.session_state.output_df.iloc[:int(len(st.session_state.output_df)*st.session_state.split_ratio)])
            learning_results_neuralfcm(st.session_state.model, total_training_samples)
        else:
            model = st.session_state.kfold_dic[fold]
            total_training_samples = len(model.train_index)
            learning_results_neuralfcm(model, total_training_samples)

    elif st.session_state.learning_algorithm == 'Particle Swarm Optimization':
        pass

def learning_results_neuralfcm(model, total_training_samples):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'Learning has finished after {np.round(np.sum(model.times), 2)} sec and {len(model.times)} epochs.\n')
            if st.session_state.epochs > len(model.times):
                st.write('Learning was terminated by the Early Stopping algorithm')
            st.write(f'Mean time per epoch: {np.round(np.mean(model.times), 3)} ms.\n')
            st.write(f'Batch size: {st.session_state.batch_size}\n')
            st.write(f'Learning was performed for **{model.fcm_iter} FCM iterations**, and with **λ = {model.l_slope}** in the sigmoid function for inference.')
            
            validation_samples = int(total_training_samples*st.session_state.validation_split)
            training_samples = total_training_samples-validation_samples
            
        with col2:
            st.write(f'**Total learning samples**: {total_training_samples},  Shuffled : **{st.session_state.shuffle}**\n')
            st.write(f'**Samples Kept for learning**: {training_samples}, and for **validating**: {validation_samples}\n')
            cl1, cl2 = st.columns(2)
            with cl1:
                st.write(f'Maximum loss: {np.round(np.max(model.history.history["loss"]), 3)}')
                st.write(f'Maximum validation loss: {np.round(np.max(model.history.history["val_loss"]), 3)}')
            with cl2:
                st.write(f'Minimum loss: {np.round(np.min(model.history.history["loss"]), 3)}\n')
                st.write(f'Minimum validation loss: {np.round(np.min(model.history.history["val_loss"]), 3)}\n')
            to_plot = st.toggle('Generate loss graph', help = 'Select this widget for plotting the curve of the learning losses')
        if to_plot:
            plot_loss(model.history)

        
def learning_results_averaged():
    '''
    The average results for the kfold option
    '''
    epochs = []
    total_times = []
    time_per_epoch = []
    max_losses = []
    min_losses = []
    max_losses_val = []
    min_losses_val = []
    for key in st.session_state.kfold_dic.keys():
        epochs.append(len(st.session_state.kfold_dic[key].times))
        total_times.append(np.sum(st.session_state.kfold_dic[key].times))
        time_per_epoch.append(np.mean(st.session_state.kfold_dic[key].times))
        max_losses.append(np.max(st.session_state.kfold_dic[key].history.history["loss"]))
        min_losses.append(np.min(st.session_state.kfold_dic[key].history.history["loss"]))
        max_losses_val.append(np.max(st.session_state.kfold_dic[key].history.history["val_loss"]))
        min_losses_val.append(np.min(st.session_state.kfold_dic[key].history.history["val_loss"]))
    
    dic = {
        'Total Epochs' : epochs,
        'Total Training Times' : total_times,
        'Times per Epoch' : time_per_epoch,
        'Max Loss Values' : max_losses,
        'Min Loss values' : min_losses,
        'Max Val Loss Values' : max_losses_val,
        'Min Val Loss values' : min_losses_val,
        }
    df = pd.DataFrame(dic, index = st.session_state.kfold_dic.keys())
    col1, col2 = st.columns(2)
    with col1:
        st.write(f'Learning required on average {np.round(np.mean(total_times), 2)} sec and {np.round(np.mean(epochs), 1)} epochs.\n')
        st.write(f'Average time per epoch: {np.round(np.mean(time_per_epoch), 3)} ms.\n')
        cl1, cl2 = st.columns(2)
        with cl1:
            st.write(f'Average max loss values: {np.round(np.mean(max_losses), 3)}\n')
            st.write(f'Average max validation loss values: {np.round(np.mean(max_losses_val), 3)}\n')
        with cl2:
            st.write(f'Average min loss values: {np.round(np.mean(min_losses), 3)}\n')
            st.write(f'Average min validation loss values: {np.round(np.mean(min_losses_val), 3)}\n')
    with col2:
        st.dataframe(df)


def plot_loss(history):
    '''
    Function to plot the loss functions of the Neural-FCM
    '''
    st.divider()
    width = st.slider("Change graph's width", 4, 10, 7)
    height = st.slider("Change graph's height", 4, 10, 5)
    fig, axs = plt.subplots(figsize = (width, height))
    axs.plot(history.history["loss"], label = 'Loss')
    axs.plot(history.history["val_loss"], label = 'Validation Loss')
    axs.set_ylabel('Neural-FCM loss')
    axs.set_xlabel('Epochs')
    axs.legend()
    st.pyplot(fig)