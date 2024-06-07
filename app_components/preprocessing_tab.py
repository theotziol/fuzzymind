import streamlit as st 


def datacleansing_widgets():
    st.subheader('Data cleansing')
    nan_values()



def nan_values():
    
    columns_na = [i for i in st.session_state.working_df.columns if st.session_state.working_df[i].isna().any()]
    
    imputation_methods = {'Interpolation':'Fill NaN values using an interpolation method',
            'Ffil':'Fill NA/NaN values by propagating the last valid observation to next valid', 
            'Bfil':'Fill NA/NaN values by using the next valid observation to fill the gap.',
            'Value':'Fill NA/NaN values with a specific value.',
            'Statistics':'Fill NA/NaN values by using a statistic based value.'}
    
    if len(columns_na) > 0:
        st.info('Delete or imputate cells and columns with missing values')
        col1, col2 = st.columns(2)
        with col1:
            with st.expander('Imputation of values'):
                column = st.selectbox('Select **column** to process', columns_na, None )
                if column is not None:
                    st.write(f'You selected the {column} column')
                method = st.selectbox('Select the **imputation method**', list(imputation_methods.keys()), None)
                if method is not None:
                    st.write(imputation_methods[method])
        with col2:
            with st.expander('Delete rows/columns'):
                delete_column = st.selectbox('Delete column', columns_na, None )
                if delete_column is not None:
                    st.write(f'You selected the {delete_column} to be deleted')
                




            



        st.text(f'Contains NA/NaN:\n\n{st.session_state.working_df.isna().any()}')
            

    else:
        st.info('No nan values were found in dataset')
        
    
        
        

def submit_method_callback(column, method):
    pass