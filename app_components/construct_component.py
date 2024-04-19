import streamlit as st
import sys
sys.path.insert(1, '../fcm_codes')
from fcm_codes.fcm_simple import *

#Some help texts for the UI widgets
help_num_concepts = 'Give an integer in the range [3, 50]'
help_iterations = 'The inference will be running for the chosen number of iterations if no equilibrium point was found.'
help_equilibrium = 'The threshold $\it{e}$ to terminate inference if the $a^{t+1}_i - a^t_i <= \it{e}$ | $\\forall a^{t+1}_i$'

#todo check for cache
def input_parameters_fcm():
    '''
    this function provides the UI widgets to manually construct an FCM.
    Returns:
        -num_concepts : int, The total number of concepts (st.number_input)
        -num_iter : int, the number of iterations (st.number_input)
        -equilibrium : float, the number to terminate inference (st.number_input)
        -inf_rule : str, one of ['kosko', 'stylios', 'rescaled']
        -activ_function: str, one of ['sigmoid', 'tanh', 'bivalent', trivalent']
    '''
    col1, col2 = st.columns(2, gap = "medium")
    with col1:
        num_concepts = st.number_input('Give the number of concepts', min_value=3, max_value=50, value = None, help = help_num_concepts)
        equilibrium = st.number_input('Give the equilibrium threshold', min_value=0.00001, max_value=0.01, value = 0.001, format="%f", help = help_equilibrium)
    with col2:
        rule_text = st.selectbox('Select the activation rule', tuple(dict_rules.keys()), index = 1)
        activation_text = st.selectbox('Select the activation function', tuple(dict_functions.keys()))
    num_iter = st.slider('Select maximum iterations', 1, 30, 20, help = help_iterations)
    return num_concepts, num_iter, equilibrium, rule_text, activation_text




