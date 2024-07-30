import streamlit as st 
import pandas as pd 
import numpy as np
import sys
from matplotlib import pyplot as plt
import time

sys.path.insert(1, '../fcm_codes')
sys.path.insert(1, '../app_components')
from fcm_codes.dynamic_fcm import *
from fcm_codes.preprocessing import split_train_test
from app_components.learning_results import *
from app_components.testing_results import *
from app_components.weight_matrix_results import *

help_learning_algorithms = "Neural-FCM is an FCM learning algorithm introduced by (Tziolas et al 2024) that utilizes a neural network for learning the FCM matrix.\
    Particle Swarm Optimization (PSO) is a population-based algorithm widely used in FCM learning tasks"

help_l_slope = "Is a parameter that affects the **convergence of FCMs** as it modifies the **steepness of the sigmoid curve**. \
    Higher values result in higher concept activation values after each inference iteration. A value < 5 is recommended for the most datasets."

help_fcm_iter_class = 'The number of **FCM iterations**. In Neural-FCM classification the FCM inference is performed for a predifined number of iterations. \
    The FCM classification literature proposes 2-5 FCM iterations.'

help_fcm_iter_timeseries = 'The number of **FCM iterations**. In Neural-FCM timeseries forecasting the FCM inference is performed for a predifined number of iterations and is associated with the timestep-ahead prediction. \
    For instance a one-step inference aims to forecast one-step ahead values. The value of the output concept at the last inference state will be used as the predicted value. \n\
        **Disabled. Equals to the timestep number in splitting parameters.**'

help_fcm_iter_regression = 'The number of **FCM iterations** for performing inference. The value of the output concept at the last inference state will be used as the predicted value.'

help_batch_size = 'The **batch size** defines the number of samples that will be propagated through the network. As **small** batch sizes require **less memory**, the maximum allowed batch size is set to 128 for better app efficiency.'

help_epochs = 'Epoch is a **complete forward pass** of **all the training data**. To avoid undertrained models, pass a high epoch number and use the early stopping algorithm.'

help_early_stopping = 'Early stopping monitors the loss in the training and validation data. \
    In case the loss in the validation data fails to improve for a predifined number of epochs (patience parameter), the training stops.'

#PSO helps
help_pso_iter = '''The maximum number of iterations to reach a good solution. Too few iterations may terminate the search prematurely.
A too large number of iterations has the consequence of unnecessary added
computational complexity (provided that the number of iterations is the only
stopping condition)'''

c1_c2 = '''\nParticles draw their strength from their cooperative nature, and are most effective
when c1 ≈ c2. While most applications use c1 = c2, the ratio between these constants is problem dependent.
If c1 >> c2, each particle is much more attracted to its own personal
best position, resulting in excessive wandering. On the other hand, if c2 >> c1,
particles are more strongly attracted to the global best position, causing particles
to rush prematurely towards optima. Low values for c1 and c2 result in smooth particle trajectories, allowing particles
to roam far from good regions to explore before being pulled back towards good
regions. High values cause more acceleration, with abrupt movement towards or
past good regions.
'''
help_c1 = f'c1 acceleration coefficient expresses how much confidence a particle has in itself. {c1_c2}'
help_c2 = f'c2 acceleration coefficient expresses how much confidence a particle has in its neighbors. {c1_c2}'
help_b = '''The inertia weight, w, controls the momentum of the particle by weighing
the contribution of the previous velocity – basically controlling how much memory of
the previous flight direction will influence the new velocity.'''

help_offset = 'Weight matrices are initialized randomly with values [-1 + x_offset, 1x-_offset].'
help_linear_decay = '''A large inertia weight of 0.9 will be initially used that will be linearly decreased to a small value 0.4, In doing so, particles
are allowed to explore in the initial search steps, while favoring exploitation as time
increases. C1 and C2 will also be modified linearly.'''
# help_loss = 'Neural-FCM loss is a triple weigthed loss of a) Categorical Cross Entropy (CCE) (weight = 2), b) Diagonal (weight = 1) and c) Output loss (weight = 1). Diagonal loss aims to push diagonal values to 0 and output loss aims to push output concept rows to 0 values'

def learning_method_widgets():
    st.subheader('Learning methods', divider = 'blue')
    col1, col2, col3 = st.columns([0.3,0.4,0.3])
    with col2:
        st.radio('Select learning algorithm', ['Neural-FCM', 'Particle Swarm Optimization'], None, help = help_learning_algorithms, key = 'learning_algorithm', horizontal=True)

    if st.session_state.learning_algorithm is not None:
        st.write(f'You selected {st.session_state.learning_algorithm} for {st.session_state.learning_task}')
        
        st.subheader('Learning parameters', divider = 'blue')
        if st.session_state.learning_algorithm == 'Neural-FCM':
            parameters_tab_neural_fcm()
        else:
            parameters_tab_pso()
    


def parameters_tab_neural_fcm():
    '''
    The parameters of Neural-FCM training
    '''
    if st.session_state.learning_task == 'Classification':
        help_fcm_iter = help_fcm_iter_class
        disabled = False
        fcm_iter = 2
    elif st.session_state.learning_task == 'Regression':
        if st.session_state.regression_split == 'Standard split':
            disabled = False
            fcm_iter = 1
            help_fcm_iter = help_fcm_iter_regression
        else:
            disabled = True
            fcm_iter = st.session_state.timestep_num
            help_fcm_iter = help_fcm_iter_timeseries
    
    with st.expander('Neural-FCM parameters...', expanded = not st.session_state.training_finished):
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.write('**Neural-FCM Classifier parameters.**')
            st.slider('λ-slope sigmoid parameter', 1, 10, 1, step = 1, key = 'l_slope', help = help_l_slope)
            st.slider('FCM iterations', 1, 10, fcm_iter, step = 1, key = 'fcm_iter', help = help_fcm_iter, disabled=disabled)
            st.radio('Batch size', [4, 16, 32, 64, 128], 3, key = 'batch_size', help = help_batch_size, horizontal=True)

        with col2:
            st.write('**Other training parameters.**')
            st.slider('Epochs', 20, 1000, 700, 5, key='epochs', help = help_epochs)
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
        train = st.button('Fit on data', on_click = _on_click_train)

    if st.session_state.train:
        st.session_state.training_finished = False
        learning()
        # st.session_state.train = False #bug when train fails it keeps on running when parameters are adjusted. It was added after training stops in each algorithm


def parameters_tab_pso():
    st.info('PSO is under construction')
    col1, col2 = st.columns(2)
    with col1:
        st.slider('Population size', 50, 500, 100, 10, key = pop_size, help = 'The particles population, or the number of matrices that will be initialized.')
        st.slider('Iterations', 50, 500, 250, 10, key = num_iter, help = help_pso_iter)
        st.slider('c1', 0.0, 2.5, 1.0, 0.1, key = c1, help = help_c1)
        st.slider('c2', 0.0, 2.5, 1.0, 0.1, key = c2, help = help_c2)
        st.slider('b (inertia weight)', 0.0, 2.5, 0.7, 0.1, key = b, help = help_b)
    with col2:
        st.slider('x offset', 0.0, 0.9, 0.2, 0.1, key = x_offset, help = help_offset)
        st.checkbox('Linear inertia decrease', True, key = linear_decay, help = help_linear_decay)


def learning():
    '''
    the function that invokes all the other functions when the train button is pressed. 
    Is not an on-click callback in order to use widgets such as st.spinner that are not working properly with on_click callbacks.
    '''
    
    if st.session_state.learning_task == 'Classification':
        if st.session_state.learning_algorithm == 'Neural-FCM':
            if st.session_state.split_method == 'KFold':
                learning_neuralfcm_classification_KFold()
            else:
                learning_neuralfcm_classification_standard()
        elif st.session_state.learning_algorithm == 'Particle Swarm Optimization':
            learning_pso_classification()
    elif st.session_state.learning_task == 'Regression':
        if st.session_state.learning_algorithm == 'Neural-FCM':
            if st.session_state.split_method == 'KFold':
                learning_neuralfcm_regression_KFold()
            else:
                learning_neuralfcm_regression_standard()
        elif st.session_state.learning_algorithm == 'Particle Swarm Optimization':
            learning_pso_regression()
    else:
        pass
        
def results_widgets():
    '''
    Generic function that gathers the results. Results are divided in three tabs 
    1. learning results (contains information regarding the training)
    2. testing results (contains information regarding the testing in unseen data)
    3. Weight matrix results (aims to plot and present the learned weight matrix)

    The widgets of each tab are invoked from another script
    '''
    if st.session_state.training_finished:
        st.subheader('Results', divider='blue')
        if st.session_state.split_method == 'KFold':
            tab_learning_results_averg, tab_testing_results_averag = st.tabs(['🎓 Average Learning Results', '📑 Average Testing Results'])
            with tab_learning_results_averg:
                learning_results(fold = 'average')
            with tab_testing_results_averag:
                testing_results(fold = 'average')
            st.divider()
            cl1, cl2, cl3 = st.columns([0.3, 0.4, 0.3])
            with cl2:
                select_fold = st.selectbox('Select fold results...', st.session_state.kfold_dic.keys(), None, format_func=lambda option: f'Fold {option}', placeholder='Choose a fold')
            if select_fold != None:
                st.caption(f'Fold {select_fold} results.')
                tab_learning_results, tab_testing_results, tab_matrix = st.tabs(['🎓 Learning Results', '📑 Testing Results', '🧮 Weight Matrix'])
                with tab_learning_results:
                    learning_results(fold = select_fold) 
                with tab_testing_results:
                    testing_results(fold = select_fold) 
                with tab_matrix:
                    weight_matrix_results(fold = select_fold) 


        else:
            tab_learning_results, tab_testing_results, tab_matrix = st.tabs(['🎓 Learning Results', '📑 Testing Results', '🧮 Weight Matrix'])
            with tab_learning_results:
                learning_results(fold = None) 
            with tab_testing_results:
                testing_results(fold = None) 
            with tab_matrix:
                weight_matrix_results(fold = None) 




#### classification learning methods
def learning_neuralfcm_classification_standard():
    with st.spinner('Learning has started. This may take a while...'):
        x_train = st.session_state.input_df.iloc[ :int(len(st.session_state.input_df)*st.session_state.split_ratio)].to_numpy()
        y_train = st.session_state.output_df.iloc[ :int(len(st.session_state.output_df)*st.session_state.split_ratio)].to_numpy()
        x_test = st.session_state.input_df.iloc[int(len(st.session_state.input_df)*st.session_state.split_ratio):].to_numpy()
        y_test = st.session_state.output_df.iloc[int(len(st.session_state.output_df)*st.session_state.split_ratio):].to_numpy()
        train_y = np.concatenate([x_train,y_train], axis = -1)
        nfcm = neural_fcm(x_train.shape[-1],y_train.shape[-1], fcm_iter=st.session_state.fcm_iter, l_slope=st.session_state.l_slope)
        nfcm.initialize_loss_and_compile('cce', lr = st.session_state.learning_rate)
        time_callback = TimeHistory()
        if st.session_state.bool_early_stopping:
            callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = st.session_state.patience, restore_best_weights=st.session_state.restore_best_weights), time_callback]
        else:
            callbacks = [time_callback]
        history = nfcm.model.fit(x_train, train_y, batch_size=st.session_state.batch_size, epochs = st.session_state.epochs, validation_split = st.session_state.validation_split, callbacks = callbacks)
        # predictions = nfcm.predict_classification(x_test)
        # nfcm.metrics_classification(y_test)
        # print(nfcm.accuracy, np.mean(time_callback.times), nfcm.f1_score_macro, nfcm.confusion_matrix)
    
        nfcm.times = time_callback.times #store the epoch times to the model
        nfcm.history = history
        st.session_state.training_finished = True
        st.session_state.train = False
        
    if st.session_state.training_finished:
        st.success('Learning has finished!')
        with st.status("Testing on unseen data...", expanded = True) as status:
            st.write('Predicting FCM weight matrices...')
            start = time.time()
            nfcm.predict_classification(x_test)
            finish = time.time()
            nfcm.metrics_classification(y_test)
            st.session_state.model = nfcm
            st.session_state.model.prediction_time = np.round(finish-start, 4)
            status.update(label = 'Testing Completed', state = 'complete', expanded = False)



def learning_neuralfcm_classification_KFold():
    from sklearn.model_selection import KFold 
    dic_kfold = {}
    with st.status('Learning has started. This may take a while...', expanded = True) as status:
        kf = KFold(n_splits = st.session_state.kfold_n_splits, shuffle = st.session_state.shuffle)
        fold = 0
        for train_index, test_index in kf.split(st.session_state.input_df):
            fold +=1
            st.write(f'Fold {fold}...')
            dic_kfold[fold] = {}
            x_train = st.session_state.input_df.iloc[train_index].to_numpy()
            y_train = st.session_state.output_df.iloc[train_index].to_numpy()
            x_test = st.session_state.input_df.iloc[test_index].to_numpy()
            y_test = st.session_state.output_df.iloc[test_index].to_numpy()
            train_y = np.concatenate([x_train,y_train], axis = -1)
            nfcm = neural_fcm(x_train.shape[-1],y_train.shape[-1], fcm_iter=st.session_state.fcm_iter, l_slope=st.session_state.l_slope)
            nfcm.initialize_loss_and_compile('cce', lr = st.session_state.learning_rate)
            time_callback = TimeHistory()
            if st.session_state.bool_early_stopping:
                callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = st.session_state.patience, restore_best_weights=st.session_state.restore_best_weights), time_callback]
            else:
                callbacks = [time_callback]
            history = nfcm.model.fit(x_train, train_y, batch_size=st.session_state.batch_size, epochs = st.session_state.epochs, validation_split = st.session_state.validation_split, callbacks = callbacks)
        
            nfcm.times = time_callback.times #store the epoch times to the model
            nfcm.history = history
            st.success(f'Learning of fold {fold} has finished!')
            start = time.time()
            nfcm.predict_classification(x_test)
            finish = time.time()
            nfcm.metrics_classification(y_test)
            nfcm.prediction_time = np.round(finish-start, 4)
            nfcm.train_index = train_index
            nfcm.test_index = test_index
            dic_kfold[fold] = nfcm         
        st.session_state.training_finished = True
        st.session_state.train = False
        st.session_state.kfold_dic = dic_kfold
        status.update(label = f'{st.session_state.kfold_n_splits}Fold has finished!', state = 'complete', expanded = False)
        
        


def learning_pso_classification():
    pass


### Regression learning methods
def learning_neuralfcm_regression_standard():
    with st.spinner('Learning has started. This may take a while...'):
        train_x, test_x, train_y, test_y = split_train_test(st.session_state.input_df.to_numpy(), st.session_state.output_df.to_numpy(), st.session_state.split_ratio)
        #callbacks
        time_callback = TimeHistory()
        if st.session_state.bool_early_stopping:
            callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = st.session_state.patience, restore_best_weights=st.session_state.restore_best_weights), time_callback]
        else:
            callbacks = [time_callback]

        #check if data have been splitted for timeseries as proposed in Neural-FCM or as simple input output scheme
        if st.session_state.timestep_num != None:
            nfcm = neural_fcm(train_x.shape[-1],output_concepts=len(st.session_state.output_columns) ,fcm_iter=st.session_state.fcm_iter, l_slope=st.session_state.l_slope)
            nfcm.initialize_loss_and_compile('mse', lr = st.session_state.learning_rate)
            history = nfcm.model.fit(train_x, train_y, batch_size=st.session_state.batch_size, epochs = st.session_state.epochs, validation_split = st.session_state.validation_split, callbacks = callbacks, shuffle = st.session_state.shuffle)
        else:
            y_train = np.concatenate([train_x,train_y], axis = -1)
            nfcm = neural_fcm(train_x.shape[-1],train_y.shape[-1], fcm_iter=st.session_state.fcm_iter, l_slope=st.session_state.l_slope)
            nfcm.initialize_loss_and_compile('mse_standard', lr = st.session_state.learning_rate)
            history = nfcm.model.fit(
                train_x, y_train, 
                batch_size=st.session_state.batch_size, 
                epochs = st.session_state.epochs, 
                validation_split = st.session_state.validation_split, 
                callbacks = callbacks, 
                shuffle = st.session_state.shuffle)
        nfcm.times = time_callback.times #store the epoch times to the model
        nfcm.history = history
        st.session_state.training_finished = True
        st.session_state.train = False

    if st.session_state.training_finished:
        st.success('Learning has finished!')
        with st.status("Testing on unseen data...", expanded = True) as status:
            st.write('Predicting FCM weight matrices...')
            start = time.time()
            nfcm.predict_regression(test_x)
            finish = time.time()
            real_array_test = st.session_state.non_norm_working_df[st.session_state.output_columns].iloc[-len(test_y):].to_numpy()
            real_array = st.session_state.non_norm_working_df[st.session_state.output_columns].to_numpy()
            nfcm.statistics_regression(real_array_test[:,0], real_array)
            nfcm.statistics_regression_norm(test_y[:, -len(st.session_state.output_columns)])
            st.session_state.model = nfcm
            st.session_state.model.prediction_time = np.round(finish-start, 4)
            status.update(label = 'Testing Completed', state = 'complete', expanded = False)


def learning_neuralfcm_regression_KFold():
    from sklearn.model_selection import KFold 
    dic_kfold = {}
    with st.status('Learning has started. This may take a while...', expanded = True) as status:
        kf = KFold(n_splits = st.session_state.kfold_n_splits, shuffle = st.session_state.shuffle)
        fold = 0
        for train_index, test_index in kf.split(st.session_state.input_df):
            fold +=1
            st.write(f'Fold {fold}...')
            dic_kfold[fold] = {}
            x_train = st.session_state.input_df.iloc[train_index].to_numpy()
            y_train = st.session_state.output_df.iloc[train_index].to_numpy()
            x_test = st.session_state.input_df.iloc[test_index].to_numpy()
            y_test = st.session_state.output_df.iloc[test_index].to_numpy()
            train_y = np.concatenate([x_train,y_train], axis = -1)
            nfcm = neural_fcm(x_train.shape[-1],y_train.shape[-1], fcm_iter=st.session_state.fcm_iter, l_slope=st.session_state.l_slope)
            nfcm.initialize_loss_and_compile('mse_standard', lr = st.session_state.learning_rate)
            time_callback = TimeHistory()
            if st.session_state.bool_early_stopping:
                callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = st.session_state.patience, restore_best_weights=st.session_state.restore_best_weights), time_callback]
            else:
                callbacks = [time_callback]
            history = nfcm.model.fit(x_train, train_y, batch_size=st.session_state.batch_size, epochs = st.session_state.epochs, validation_split = st.session_state.validation_split, callbacks = callbacks)
        
            nfcm.times = time_callback.times #store the epoch times to the model
            nfcm.history = history
            st.success(f'Learning of fold {fold} has finished!')
            start = time.time()
            nfcm.predict_regression(x_test)
            finish = time.time()

            real_array_test = st.session_state.non_norm_working_df[st.session_state.output_columns].iloc[test_index].to_numpy()
            real_array = st.session_state.non_norm_working_df[st.session_state.output_columns].to_numpy()
            nfcm.statistics_regression(real_array_test[:,0], real_array)
            nfcm.statistics_regression_norm(y_test[:, -len(st.session_state.output_columns)])

            nfcm.prediction_time = np.round(finish-start, 4)
            nfcm.train_index = train_index
            nfcm.test_index = test_index
            dic_kfold[fold] = nfcm         
        st.session_state.training_finished = True
        st.session_state.train = False
        st.session_state.kfold_dic = dic_kfold
        status.update(label = f'{st.session_state.kfold_n_splits}Fold has finished!', state = 'complete', expanded = False)


def learning_pso_regression():
    pass



def _on_click_train():
    ###check if normalized dataset exists, if not, assign non_normalized to working_df to avoid bugs when non-normalized is called (regression)
    if type(st.session_state.non_norm_working_df) == type(None):
        st.session_state.non_norm_working_df = st.session_state.working_df.copy() 
    st.session_state.train = True
    st.session_state.training_finished = False


