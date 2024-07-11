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



def learning_results():
    '''
    the function that invokes all the other functions when the learning is finished. 
    It aims to gather and show results from learning. 
    '''
    
    
    if st.session_state.learning_algorithm == 'Neural-FCM':
        if st.session_state.split_method == 'KFold':
            pass
        else:
            st.caption('Learning results.')
            learning_results_neuralfcm_standard()
    elif st.session_state.learning_algorithm == 'Particle Swarm Optimization':
        pass

def learning_results_neuralfcm_standard():
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'Learning has finished after {np.round(np.sum(st.session_state.model.times), 2)} sec and {len(st.session_state.model.times)} epochs.\n')
            if st.session_state.epochs > len(st.session_state.model.times):
                st.write('Learning was terminated by the Early Stopping algorithm')
            st.write(f'Mean time per epoch: {np.round(np.mean(st.session_state.model.times), 3)} ms.\n')
            st.write(f'Batch size: {st.session_state.batch_size}\n')
            st.write(f'Learning was performed for **{st.session_state.model.fcm_iter} FCM iterations**, and with **λ = {st.session_state.model.l_slope}** in the sigmoid function for inference.')
            total_training_samples = len(st.session_state.output_df.iloc[:int(len(st.session_state.output_df)*st.session_state.split_ratio)])
            validation_samples = int(total_training_samples*st.session_state.validation_split)
            training_samples = total_training_samples-validation_samples
            
        with col2:
            st.write(f'**Total learning samples**: {total_training_samples},  Shuffled : **{st.session_state.shuffle}**\n')
            st.write(f'**Samples Kept for learning**: {training_samples}, and for **validating**: {validation_samples}\n')
            cl1, cl2 = st.columns(2)
            with cl1:
                st.write(f'Maximum loss: {np.round(np.max(st.session_state.model.history.history["loss"]), 3)}')
                st.write(f'Maximum validation loss: {np.round(np.max(st.session_state.model.history.history["val_loss"]), 3)}')
            with cl2:
                st.write(f'Minimum loss: {np.round(np.min(st.session_state.model.history.history["loss"]), 3)}\n')
                st.write(f'Minimum validation loss: {np.round(np.min(st.session_state.model.history.history["val_loss"]), 3)}\n')
            to_plot = st.toggle('Generate loss graph', help = 'Select this widget for plotting the curve of the learning losses')
        if to_plot:
            plot_loss(st.session_state.model.history)
        


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