import streamlit as st 
import numpy as np 
import pandas as pd
from pandas.api.types import is_numeric_dtype 


def transformation_widgets():
    st.subheader('Data Transformation', divider = 'blue')
    
    cl1, cl2, cl3 = st.columns([0.3, 0.4, 0.3])
    with cl2:
        enc_method = st.radio('Select the encoding method', ['One-hot encoding', 'Integer encoding'], index = None, horizontal = True)

    st.write('')
    if enc_method == 'One-hot encoding':
        st.write('Encode categorical features as a [one-hot](https://en.wikipedia.org/wiki/One-hot) numeric array.')
        with st.expander('Text to numerical (One-hot encoding)...'):
            one_hot_column = st.selectbox('Select column to encode.', st.session_state.working_df.columns, None, placeholder='Select column...')
            if one_hot_column is not None:
                #warn for numeric columns, but allow it for one hot encoding
                if is_numeric_dtype(st.session_state.working_df[one_hot_column]):
                    st.warning('You selected a numeric column. Verify that numeric values are Integers', icon = '⚠️')
                df = column_to_categorical(one_hot_column, st.session_state.working_df)
                cl1, cl2 = st.columns(2)   
                with cl1:
                    st.caption(':blue[Current dataset.]')
                    st.dataframe(st.session_state.working_df, hide_index=True)
                with cl2:
                    st.caption(':red[Encoded dataset.]')
                    st.dataframe(df, hide_index=True)
                submit = st.button('Submit changes', on_click=submit_one_hot, args = (df, ))
                if submit:
                    st.success('Transformation was succesfully applied.')
            

    elif enc_method == 'Integer encoding': 
        st.write('Transform columns containing text values into columns containing integers.')
        with st.expander('Text to numerical (Integer Encoding)...'):
            text_dtypes = ('object', 'bool')
            columns = [col for col in st.session_state.working_df.columns if st.session_state.working_df[col].dtype.name in text_dtypes]
            if len(columns) > 0:
                column = st.selectbox('Select a column that contains text', columns, None, placeholder='Select column...')
                if column is not None:
                    df, replace_dic = text_to_numerical(column, st.session_state.working_df)
                    st.caption(f'Column {column} unique values and replacement information')
                    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
                    with col2:
                        st.dataframe(df, hide_index=True)
                    replace = st.button('Replace', key = 'replace_text', on_click=submit_data_transformation, args = (column, replace_dic))
                    if replace:
                        st.success('Transformation was succesfully applied.')

            else:
                st.info('No column with text was found in this dataset...')
    else:
        st.write('Select an encoding method.')
    



@st.cache_data
def text_to_numerical(column, working_df):
            uniques, counts = np.unique(working_df[column], return_counts=True)
            numbers = [i for i in range(len(uniques))]
            dic = {
                'Unique values' : uniques,
                'Count' : counts,
                'To be replaced with numbers' : numbers
            }
            replace_dic = dict(zip(uniques, numbers))
            df = pd.DataFrame(dic)
            return df, replace_dic
            

@st.cache_data
def column_to_categorical(column, working_df):
    df = working_df.copy()
    labels = np.unique(df[column])
    zeros = np.zeros(len(df))
    for i in range(len(labels)):
        df[labels[i]] = zeros
        df[labels[i]][df[column] == labels[i]] = 1
    df.pop(column)
    return df

    


### on click callbacks
def submit_data_transformation(column, replace_dic):
    st.session_state.working_df[column].replace(replace_dic, inplace = True)
    st.session_state.working_df[column] = st.session_state.working_df[column].astype('int16')
    st.session_state.changed = True


def submit_one_hot(encoded_df):
    st.session_state.working_df = encoded_df
    st.session_state.changed = True
    