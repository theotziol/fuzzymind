import streamlit as st 
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import io


def datacleansing_widgets():
    st.subheader('Data cleansing')
    nan_values()



def nan_values():
    columns_na = [i for i in st.session_state.working_df.columns if st.session_state.working_df[i].isna().any()]
    df = st.session_state.working_df.copy()
    imputation_methods = {'Interpolation':'Fill NaN values using an interpolation method',
            'Ffil':'Fill NA/NaN values by propagating the last valid observation to next valid', 
            'Bfil':'Fill NA/NaN values by using the next valid observation to fill the gap.',
            'Value':'Fill NA/NaN values with a specific value.',
            'Statistics':'Fill NA/NaN values by using a statistic based value.'}
    
    if len(columns_na) > 0:
        with st.expander('Imputation...', len(columns_na) > 0):
            st.info('Delete or imputate cells and columns with missing values')
            column = st.selectbox('Select **column** to process', columns_na, None )
            if column is not None:
                st.write(f'You selected the {column} column')
            method = st.selectbox('Select the **imputation method**', list(imputation_methods.keys()), None, key='fill_method')
            if method is not None:
                fig, axs = plt.subplots(figsize = (12, 4))
                st.write(imputation_methods[method])
                if method == 'Value':
                    value = value_imputation_widgets()
                    df['imputed_values'] = df[column].fillna(value)
                    df.plot(y= ['imputed_values', column],ax = axs)
                    st.pyplot(fig)
                    submit = st.button('Submit', key = 'submit_val', on_click=submit_imputation_value, args=(column, value,))
                    
                elif method == 'Statistics':
                    value = statistics_imputation_widgets(column)
                    df['imputed_values'] = df[column].fillna(value)
                    df.plot( y= ['imputed_values', column],ax = axs)
                    st.pyplot(fig)
                    submit = st.button('Submit', key = 'submit_stats', on_click=submit_imputation_value, args=(column, value))

                elif method == 'Interpolation':
                    interp_method = interpolation_imputation_widgets()
                    df['imputed_values'] = df[column].interpolate(interp_method)
                    df.plot(y= ['imputed_values', column],ax = axs)
                    st.pyplot(fig)
                    submit = st.button('Submit', key = 'submit_inter', on_click=submit_interpolation_value, args=(column, interp_method))
                elif method == 'Ffil':
                    df['imputed_values'] = df[column].ffill()
                    df.plot( y= ['imputed_values', column],ax = axs)
                    st.pyplot(fig)
                    submit = st.button('Submit', key = 'submit_ffil', on_click=submit_ffil, args=(column,))
                else:
                    df['imputed_values'] = df[column].bfill()
                    df.plot(y= ['imputed_values', column],ax = axs)
                    st.pyplot(fig)
                    submit = st.button('Submit', key = 'submit_bfil', on_click=submit_bfil, args=(column,))
                

        with st.expander('Delete rows/columns'):
            col1, col2 = st.columns(2)
            with col1:
                delete_column = st.selectbox('Delete a column with NaN values', columns_na, None )
                if delete_column is not None:
                    st.write(f'You selected the {delete_column} to be deleted')
                    delete_col_button = st.button('Submit', key = 'del_col', on_click=submit_deletion_col, args = (delete_column,))
            with col2:
                st.write('Dischard **all rows** with missing values.')
                delete_all_rows_button = st.button('Dischard rows', key = 'del_rows', on_click=submit_deletion_rows)


        
            

    else:
        st.info('No nan values were found in dataset')
        col1,col2,col3 = st.columns((0.3,0.4,0.3))
        with col2:
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        
    
def value_imputation_widgets():
    '''
    This function is used for providing widgets when the users decides to manually provide a value for imputation.
    Returns either a numeric or a text value
    '''
    bool_num = st.toggle('Numeric value', True, help='Deactivate if text will be given')
    
    if bool_num:
        value = st.number_input('Give a numeric value', 0.0)
    else:
        value = st.text_input('Provide a string type value',)
    return value

def statistics_imputation_widgets(column):
    '''
    This function is used for providing widgets when the users decides to manually provide a value for imputation.
    Returns either a numeric or a text value
    '''
    
    try:
        mean = st.session_state.working_df[column].mean()
        median = st.session_state.working_df[column].median()
        maxx = st.session_state.working_df[column].max()
        minn = st.session_state.working_df[column].min()
        dic = {
        'Mean' :mean,
        'Median' :median,
        'Max' : maxx,
        'Min' : minn
        }
        st.text(dic)
        selection = st.selectbox('Select statistic method for imputation', list(dic.keys()), )
        
        return dic[selection]
    except Exception as e:
        st.warning(e)
        return None

    
def interpolation_imputation_widgets():
    '''
    This function is used for providing widgets when the users decides to manually provide a value for imputation.
    Returns either a numeric or a text value
    '''
    
    methods = {
        'Linear' : 'linear',
        'Nearest' : 'nearest',
        # 'Cubic' : 'cubic',
        # 'Quadratic' : 'quadratic',
    }
    selection = st.selectbox('Select method for imputation', list(methods.keys()))
    return methods[selection]
        
        
        
### submit callbaks for on click events
def submit_imputation_value(column, value):
    st.session_state.working_df[column].fillna(value, inplace = True)
    # st.success(f'NaN cells were succesfully imputed in {column} with the value "{value}."')
    st.session_state['fill_method'] = None

def submit_interpolation_value(column, method):
    st.session_state.working_df[column].interpolate(method, inplace = True)
    # st.success(f'NaN cells were succesfully imputed in {column} with the "{method}" interpolation method.')
    st.session_state['fill_method'] = None

def submit_ffil(column):
    st.session_state.working_df[column].ffill(inplace = True)
    # st.success(f'NaN cells were succesfully imputed in {column} by propagating forward the previous values.')
    st.session_state['fill_method'] = None

def submit_bfil(column):
    st.session_state.working_df[column].bfill(inplace = True)
    # st.success(f'NaN cells were succesfully imputed in {column} by propagating forward the previous values.')
    st.session_state['fill_method'] = None


def submit_deletion_rows():
    st.session_state.working_df.dropna(inplace = True)


def submit_deletion_col(column):
    st.session_state.working_df.drop([column], axis = 1, inplace=True)




