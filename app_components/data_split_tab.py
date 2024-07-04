import streamlit as st 



help_splitting_method = "The **Standard** method allows the user to use their own desired ratio to split the dataset. The first part will be used for training and the last will be used for testing.\
    \nThe **KFold** cross-validation method splits the dataset into k consecutive folds. Each fold is then used once as a test while the k - 1 remaining folds form the training set."

def spliting_widgets():
    st.subheader('Data Split', divider = 'gray')
    st.selectbox('Select the splitting method', ['Standard', 'KFold'], key = 'split_method', help = help_splitting_method)
    #warn for kfold and timeseries
    if st.session_state.split_method == 'KFold':
        if st.session_state.learning_task == 'Regression':
            st.warning('You selected KFold with a regression task, this may not work properly with timeseries data', icon = '⚠️')
        col1, col2 = st.columns(2)
        with col1:
            st.radio('Number of splits', [5, 10], key = 'kfold_n_splits', horizontal=True)
            st.write(f'Learning will be performed with **{st.session_state.kfold_n_splits}Fold cross-validation**. A {100//st.session_state.kfold_n_splits}% chunk of data will be used for **testing**')
        with col2:
            st.checkbox('Shuffle dataset', True, key = 'kfold_shuffle', help = 'Select indices at random')

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.slider('Select splitting ratio', 0.6, max_value=0.9, value = 0.8, step = 0.05, key='split_ratio')
            st.write(f'{int(100*st.session_state.split_ratio)}% will be used for **training** and {100 - int(100*st.session_state.split_ratio)}% will be used for **testing**.')
        with col2:
            st.checkbox('Shuffle dataset', True, key = 'standard_shuffle', help = 'Shuffle dataset prior to splitting')
            if st.session_state.learning_task == 'Regression' and st.session_state.standard_shuffle:
                st.warning('You selected shuffle with a regression task, this may not work properly with timeseries data', icon = '⚠️')





def split_input_target():
    st.subheader('Split input-output columns', divider = 'gray')
    st.multiselect('Select output column(s)', st.session_state.working_df.columns, placeholder='Choose column(s)...', key = 'output_columns')
    st.write(f'You selected {st.session_state.output_columns} as output columns...')







            
            
        

    
    