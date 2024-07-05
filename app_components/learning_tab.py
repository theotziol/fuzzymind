import streamlit as st 
import pandas as pd 


help_learning_algorithms = "Neural-FCM is an FCM learning algorithm introduced by (Tziolas et al 2024) that utilizes a neural network for learning the FCM matrix.\
    Particle Swarm Optimization (PSO) is a population-based algorithm widely used in FCM learning tasks"

help_l_slope = "Is a parameter that affects the **convergence of FCMs** as it modifies the **steepness of the sigmoid curve**. \
    Higher values result in higher concept activation values after each inference iteration. A value < 5 is recommended for the most datasets."

help_fcm_iter_class = 'The number of FCM iterations. In Neural-FCM classification the FCM inference is performed for a predifined number of iterations. \
    The FCM classification literature proposes 2-5 FCM iterations.'

help_fcm_iter_timeseries = 'The number of FCM iterations. In Neural-FCM timeseries forecasting the FCM inference is performed for a predifined number of iterations and is associated with the timestep-ahead prediction. \
    For instance a one-step inference aims to forecast one-step ahead values.'

help_fcm_iter_regression = 'The number of FCM iterations.'

help_batch_size = 'The batch size defines the number of samples that will be propagated through the network. As **small batch sizes** require **less memory**, the maximum allowed batch size is set to 128 for better app efficiency.'

help_epochs = 'Epoch is a **complete forward pass** of **all the training data**. To avoid undertrained models, pass a high epoch number and use the early stopping algorithm.'

help_early_stopping = 'Early stopping monitors the loss in the training and validation data. \
    In case the loss in the validation data fails to improve for a predifined number of epochs (patience parameter), the training stops.'

def learning_method_widgets():
    st.subheader('Learning methods', divider = 'blue')
    col1, col2, col3 = st.columns([0.3,0.4,0.3])
    with col2:
        st.radio('Select learning algorithm', ['Neural-FCM', 'Particle Swarm Optimization'], None, help = help_learning_algorithms, key = 'learning_algorithm', horizontal=True)

    if st.session_state.learning_algorithm is not None:
        st.write(f'You selected {st.session_state.learning_algorithm} for {st.session_state.learning_task}')
        tab_param, tab_results = st.tabs(['📓 Training parameters', '📋 Results'])
        with tab_param: 
            if st.session_state.learning_algorithm == 'Neural-FCM':
                parameters_tab_neural_fcm()
            else:
                parameters_tab_pso()
        with tab_results:
            if not st.session_state.training_finished:
                st.markdown(
                    """
                    No training has been performed yet. Use the **📓 Training parameters** tab to initialize training.
                    # ⛔ This tab will be accesible after data training finishes. 
                    """
                    )
    



def parameters_tab_neural_fcm():
    if st.session_state.learning_task == 'Classification':
        help_fcm_iter = help_fcm_iter_class
        fcm_iter = 2
    elif st.session_state.learning_task == 'Timeseries forecasting':
        help_fcm_iter = help_fcm_iter_timeseries
        fcm_iter = 1
    else:
        help_fcm_iter = help_fcm_iter_regression
        fcm_iter = 2
    
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.write('**Neural-FCM Classifier parameters.**')
        st.slider('λ-slope sigmoid parameter', 1, 10, 1, step = 1, key = 'l_slope', help = help_l_slope)
        st.slider('FCM iterations', 1, 10, fcm_iter, step = 1, key = 'fcm_iter', help = help_fcm_iter_class)
        st.radio('Batch size', [4, 16, 32, 64, 128], 3, key = 'batch_size', help = help_batch_size, horizontal=True)


    with col2:
        st.write('**Other training parameters.**')
        st.slider('Epochs', 20, 800, 500, 5, key='epochs', help = help_epochs)
        st.checkbox('Early stopping', True, key = 'bool_early_stopping', help = help_early_stopping)
        if st.session_state.bool_early_stopping:
            cl1, cl2 = st.columns(2)
            with cl1:
                st.slider('Epochs patience', 10, 40, 20, 1, key = 'patience', help = 'The number of epochs that the early stopping algorithm shall wait before stopping the training.')
            with cl2:
                st.write('')
                st.checkbox('Restore best weights', True, key='restore_best_weights', help = 'Whether to acquire the weights of the epoch where the minimum validation error was occured or at the final epoch (after patience epochs)')
        st.slider('Learning rate', 0.0001, 0.01, 0.001, 0.0001, format = '%0.4f', key='learning_rate', help = 'The learning rate for the [Adam](https://keras.io/api/optimizers/adam/) optimizer. Recommended 0.001, (Adam default)')
    
    c1, c2, c3 = st.columns([0.4, 0.3, 0.3])
    with c2:
        ## to do initialize training
        train = st.button('Train')

            

    
            

def parameters_tab_pso():
    st.info('PSO is under construction')
