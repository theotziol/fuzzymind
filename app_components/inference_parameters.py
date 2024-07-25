import streamlit as st
import sys
sys.path.insert(1, '../fcm_codes')
from fcm_codes.general_functions import *

#Some help texts for the UI widgets

help_iterations = 'The inference will be running for the chosen number of iterations if no equilibrium point was found.'
help_equilibrium = 'The threshold $\it{h}$ to terminate inference if the $a^{t+1}_i - a^t_i <= \it{h}$ | $\\forall a^{t+1}_i$'
help_l = 'λ is a parameter in sigmoid function that affects the steepness of the curve'
help_b = 'b is a parameter in sigmoid function that affects the shifting of the curve'
sigmoid_formula = '$f(x)=\\dfrac{1}{1+e^{-(λx) + b}}$'
tanh_formula = '$f(x)=\\dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$'
bivalent_formula = '$f(x)= \\begin{cases} 1, & x > 0\\\ 0, & x \leq 0 \end{cases}$'
trivalent_formula = '$f(x)= \\begin{cases} -1, & x < 0\\\ 0, & x = 0\\\ 1, & x > 1 \end{cases}$'

kosko_formula = '$A^{t+1}_i = f(\\sum_{j = 1, \\\ i \\neq j}^M w_{ji} A^{t}_j)$'
modified_kosko_formula = '$A^{t+1}_i = f(\\sum_{j = 1, \\\ i \\neq j}^M w_{ji} A^{t}_j + A^{t}_i)$'
rescaled_formula = '$A^{t+1}_i = f(\\sum_{j = 1, \\\ i \\neq j}^M w_{ji} (2A^{t}_j - 1) + (2A^{t}_i -1)$'

#todo check for cache
def inference_parameters():
    '''
    this function provides the UI widgets to manually construct an FCM.
    Returns:
        -num_iter : int, the number of iterations (st.number_input)
        -equilibrium : float, the number to terminate inference (st.number_input)
        -inf_rule : str, one of ['kosko', 'stylios', 'rescaled']
        -activ_function: str, one of ['sigmoid', 'tanh', 'bivalent', trivalent']
    '''
    col1, col2 = st.columns(2, gap = "medium")
    with col1:
        num_iter = st.slider('Select maximum iterations', 1, 40, 20, help = help_iterations)
        equilibrium = st.number_input('Give the equilibrium threshold', min_value=0.00001, max_value=0.01, value = 0.001, format="%f", help = help_equilibrium)
        rule_text = st.selectbox('Select the activation rule', rules, index = 1)
        if rule_text == 'Kosko':
            st.write(kosko_formula)
        if rule_text == 'Modified Kosko':
            st.write(modified_kosko_formula)
        if rule_text == 'Rescaled':
            st.write(rescaled_formula)
        
    with col2:
        activation_text = st.selectbox('Select the activation function', functions)
        if activation_text == 'Sigmoid':
            st.write(sigmoid_formula)
            l = st.number_input('λ sigmoid parameter', min_value=1, max_value=10, value = 1, help = help_l)
            b = st.number_input('b sigmoid parameter', min_value=-10, max_value=10, value = 0, help = help_b)
        else:
            l, b = None, None
        
        if activation_text == 'Tanh':
            st.write(tanh_formula)
        if activation_text == 'Bivalent':
            st.write(bivalent_formula)
        if activation_text == 'Trivalent':
            st.write(trivalent_formula)

    return num_iter, equilibrium, rule_text, activation_text, l , b




