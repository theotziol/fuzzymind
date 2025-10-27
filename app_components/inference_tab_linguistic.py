import streamlit as st
import sys

sys.path.insert(1, "../fcm_codes")
from fcm_codes.general_functions import *


def non_matrix_definition():
    """
    The message to be shown if a weight matrix has not be defined
    """
    st.markdown(
        """
        Use the **FCM Design** 👆 tab to construct an FCM either manually,
        or by uploading file that contains a weight matrix for FCM definition.
        # ⛔ This tab will be accesible after the FCM construction.  
        """
    )


def define_initial_values(edited_matrix, activation):
    """
    The component that is used for creating a data_editor widget to define the initial concept vector
    Args:
        edited_matrix : a pd.DataFrame that contains the weight matrix
        rule: string, the selected activation rule (formula) it is used to define a config file, prompting the user of the correct initial concept values
    """

    st.subheader("Define the initial concept state vector", divider="blue")
    state_df, configs = create_initial_state_vector(edited_matrix.columns, activation)
    initial_state_vector = st.data_editor(state_df, column_config=configs)
    return initial_state_vector


def inference_results(df_inference_process):
    inference_placeholder = st.empty()
    with inference_placeholder.container():
        st.subheader("Inference Results", divider="blue")
        st.caption("Inference Dataframe")
        st.dataframe(df_inference_process)
        st.caption("Inference Graph")
        st.line_chart(df_inference_process)

    return inference_placeholder
