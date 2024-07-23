import streamlit as st 
from numpy import concatenate
from pandas import DataFrame
from copy import deepcopy as dc

help_regression_split = """
**Standard Split:** 
Standard split, as the name suggests, splits the input columns from the output column(s). 
Using an FCM approach this means that the input variables or (independent variables) are employed as input concepts and influence the output (dependent variable). \n\n 

**Timestep split:**
The split used by Neural-FCM (Tziolas et al. 2024) for **timeseries forecasting**. 
This timeseries forecasting splitting method employs the vectors of previous timesteps (both dependent and independent variables) for predicting the future timesteps. 
In an FCM approach this means thet the input concepts are the values of both independent and dependent variables at $\\it{\\textbf{x}^{t=0}}$ and the output is the concepts' state at the timestep $\\it{x^{t}}$. 
Where $\\it{t}$ represents also the pre-defined FCM iterations.\n

During **KFold** or when **shuffle** is enabled only standard split is performed. 

"""

help_timestep = 'The number $\\it{t}$ so the input row vector $\\it{\\textbf{x}^t}$ will kept as input (independent variables) and the $\\it{\\textbf{x}^{t+t}}$ will become the output vector (dependent variables)'

help_splitting_method = "The **Standard** method allows the user to use their own desired ratio to split the dataset. The first part will be used for training and the last will be used for testing.\
    \nThe **KFold** cross-validation method splits the dataset into k consecutive folds. Each fold is then used once as a test while the k - 1 remaining folds form the training set."



def spliting_widgets():
    st.subheader('Data Split', divider = 'gray')
    method = st.selectbox('Select the splitting method', ['Standard', 'KFold'], None, key = 'split_method', help = help_splitting_method, on_change=_widgets_on_change)
    #warn for kfold and timeseries
    if st.session_state.split_method == 'KFold':
        col1, col2 = st.columns(2)
        with col1:
            st.radio('Number of splits', [5, 10], key = 'kfold_n_splits', horizontal=True)
            st.write(f'Learning will be performed with **{st.session_state.kfold_n_splits}Fold cross-validation**. A {100//st.session_state.kfold_n_splits}% chunk of data will be used for **testing**')
        with col2:
            st.checkbox('Shuffle dataset', st.session_state.learning_task != 'Regression', key = 'shuffle', help = 'Select indices at random')
            st.slider('Validation split', 0.1, 0.3, 0.2, 0.05, key = 'validation_split', on_change=_widgets_on_change, help = 'The proportion of the training dataset that is kept for validating training (validation dataset)')

    elif st.session_state.split_method == 'Standard':
        col1, col2 = st.columns(2)
        with col1:
            st.slider('Select splitting ratio', 0.6, max_value=0.9, value = 0.8, step = 0.05, key='split_ratio', on_change=_widgets_on_change)
            st.write(f'{int(100*st.session_state.split_ratio)}% will be used for **training** and {100 - int(100*st.session_state.split_ratio)}% will be used for **testing**.')
        with col2:
            st.checkbox('Shuffle dataset', st.session_state.learning_task != 'Regression', key = 'shuffle', help = 'Shuffle dataset prior to splitting.')
            st.slider('Validation split', 0.1, 0.3, 0.2, 0.05, key = 'validation_split', help = 'The proportion of the training dataset that is kept for validating training (validation dataset)', on_change=_widgets_on_change)
    
    if method != None:   
        split_input_target()
    else:
        st.warning('Select a training/testing splitting method to continue...')



def split_input_target():
    st.subheader('Split input-output columns', divider = 'gray')
    if st.session_state.learning_task == 'Classification':
        st.info('Neural-FCM classifier requires categorical (one-hot-encoded) outputs as Categorical Cross Entropy is employed for learning.', icon='ℹ️')

    st.multiselect('Select output column(s)', st.session_state.working_df.columns, placeholder='Choose column(s)...', key = 'output_columns', on_change=_widgets_on_change)
    if len(st.session_state.output_columns) == 0:
        st.warning(f'Select output column(s) to continue...')
    else:
        st.write(f'You selected {st.session_state.output_columns} as output columns...')
        # in regression ask for a splitting method
        if st.session_state.learning_task == 'Regression': 
            reggresion_split()
        else:   
            st.select_slider('Default output value', [0.0, 0.5, 1.0], 0.5, key = 'default_output_value', help = 'FCM construction requires an initial (dummy) value for the output concepts.', on_change=_widgets_on_change)
            st.session_state.timestep_num = None

        submit = st.button('Submit', on_click = submit_splitting_parameters_callback)
        
        if submit:
            st.toast('Submitted splitting parameters', icon = '✔️')
            st.success('You can now proceed to the **🧠 Learning tab** on top 👆, to initialize a learning scheme!')
            col1, col2 = st.columns(2)
            with col1:
                st.caption('Input dataframe.')
                st.dataframe(st.session_state.input_df)
            with col2:
                st.caption('Output dataframe.')
                st.dataframe(st.session_state.output_df)



def reggresion_split():
    '''
    widgets for splitting input/output during regression.
    is is being invoked in split_input_target.
    '''
    if st.session_state.split_method == 'KFold' or st.session_state.shuffle:
        disabled = True
    else:
        disabled = False
    cl1, cl2, cl3 = st.columns([0.3, 0.4, 0.3])
    with cl2:
        st.radio('Select the regression splitting method', ['Standard split', 'Timestep split'], 0, 
        key = 'regression_split', 
        help = help_regression_split, 
        on_change=_widgets_on_change, 
        disabled=disabled)
        if st.session_state.regression_split == 'Timestep split':
            st.slider('Select timestep number', 1, 8, 1, 1, key = 'timestep_num', help = help_timestep, on_change=_widgets_on_change)
        else:
            st.select_slider('Default output value', [0.0, 0.5, 1.0], 0.5, key = 'default_output_value', help = 'FCM construction requires an initial (dummy) value for the output concepts.')
            st.session_state.timestep_num = None



def submit_splitting_parameters_callback():
    st.session_state.initialized_preprocessing = True
    st.session_state.training_finished = False
    
    # st.session_state.input_df = st.session_state.working_df[[i for i in st.session_state.working_df.columns if i not in st.session_state.output_columns]]
    # st.session_state.output_df = st.session_state.working_df[st.session_state.output_columns]
    if st.session_state.timestep_num == None:
        st.session_state.input_df, st.session_state.output_df = _split_labels(st.session_state.working_df, st.session_state.output_columns, st.session_state.default_output_value, st.session_state.shuffle)
    else:
        st.session_state.input_df, st.session_state.output_df = _split_input_target_timeseries(st.session_state.working_df, st.session_state.timestep_num)


        
### Preprocessing methods. 
## To-do: Consider moving them in a preprocessing script
def _split_labels(df, labels , value = 0.5, shuffle = True):
    '''
    function for classification tasks. 
    It splits the original dataframe into 1) input_df and 2) labels_df. 
    FCM classification requires the input vector to have the shape of inputs+outputs.
    Thus, this function separates the original label values and replaces them with the dummy {value}.
    Args:
        df: the pandas dataframe
        labels : list. the column names.
        value : float. the dummy value that will be given to the input vector for the labels (default = 0.5)
        shuffle : Boolean. whether to shuffle df (Default = True)  
    Returns:
        input_df : pandas dataframe
        df_labels : pandas dataframe
    '''
    if shuffle:
        input_df = df.copy().sample(frac=1).reset_index(drop=True)
    else:
        input_df = df.copy()
    df_labels = input_df[labels]
    input_df[labels] = value
    return input_df, df_labels

    
def _split_input_target_timeseries(df, timesteps = 1):
    '''
    Regression function for Neural-FCM
    Recieves a df and splits it into input output
    Args:
        df : type pd.DataFrame
        timesteps: int, the time distance to predict
    Returns:
        x, y numpy arrays
    '''
    y = df.iloc[timesteps:]
    x = df.iloc[:-timesteps]
    y = concatenate([x.to_numpy(),y.to_numpy()], axis = -1)
    if timesteps == 1:
        timestep_word = 'timestep'
    else:
        timestep_word = 'timesteps'
    columns = concatenate([x.columns, [i + f' after {timesteps} {timestep_word}' for i in x.columns]])
    y = DataFrame(y, columns = columns)

    return x,y



def _widgets_on_change():
    '''
    This is a callback method that changes the training state to avoid crushing the results, when a train has already implemented
    '''
    st.session_state.training_finished = False


    


