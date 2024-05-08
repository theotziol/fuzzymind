# This file contains the fcm code for inference
import numpy as np 
import pandas as pd 


class FCM_numpy:
    def __init__(
        self,
        initial_state,
        weight_matrix,
        n_iterations = 20,
        e = 0.001,
        activation_function = 'Sigmoid',
        inference_rule = 'Modified Kosko',
        l = 1,
        b = 0,
        ):
        '''
        FCM constructor
        This is a numpy implementation of the FCM inference.
        Args:
            initial_state: pd.DataFrame that contains the initial concept values
            weight_matrix: pd.DataFrame that contains the weight_matrix
            n_iterations: int, the maximum number of iterations to perform FCM inference if no stopping condition is met
            e: float, the equilibrium threshold to terminate inference. 
            activation_function: str, one of ['Sigmoid', 'Tanh', 'Bivalent', 'Trivalent']
            inference_rule: str, one of ['Kosko', 'Modified Kosko', 'Rescaled']
            l: int, sigmoid slope. An optional parameter to be passed to the sigmoid function to affect the steepness
            b: int, sigmoid shift. An optional parameter to be passed to the sigmoid function to affect the shifting of the curve
        '''

        self.initial_state = initial_state.to_numpy()
        self.columns = weight_matrix.columns
        self.weight_matrix = weight_matrix.to_numpy()
        self.n_iterations = n_iterations
        self.e = e
        self.activation_function = activation_function
        self.inference_rule = inference_rule
        self.l = l
        self.b = b
        self.__initialize_inference_parameters()


    def inference(
        self,
        ):
        '''
        The inference function. 
        Returns:
            A pd.DataFrame of the inference process
        '''
        stopping_conditions = False
        self.inference_process = [
            self.initial_state
        ]
        while not stopping_conditions:
            new_state = self.formula(self.inference_process[-1])
            fixed_new_state = self.function(new_state)
            self.inference_process.append(fixed_new_state)
            stopping_conditions = self.__check_stoping_conditions()
        inference_process = np.array(self.inference_process)
        df = pd.DataFrame(inference_process[:, 0, :], columns = self.columns)
        return df.round(4)
        

    def __initialize_inference_parameters(
        self,
        ):
        if self.inference_rule == 'Modified Kosko':
            self.formula = self._modified_kosko
        elif self.inference_rule == 'Rescaled':
            self.formula = self._rescaled
        else:
            self.formula = self._kosko
        
        if self.activation_function == 'Bivalent':
            self.function = bivalent
            
        elif self.activation_function == 'Tanh':
            self.function = tanh

        elif self.activation_function == 'Trivalent':
            self.function = trivalent
        
        else:
            self.function = lambda x: sigmoid(x, self.l, self.b)
        
    def __check_stoping_conditions(
        self,
        ):
        iteration = len(self.inference_process)
        dif = np.abs(self.inference_process[-1] - self.inference_process[-2])
        max_dif = dif.max()
        if iteration > self.n_iterations or max_dif <= self.e:
            return True
        else:
            return False


    def _kosko(
        self,
        state_vector
        ):
        return np.matmul(state_vector, self.weight_matrix)

    def _modified_kosko(
        self,
        state_vector
        ):
        return state_vector + np.matmul(state_vector, self.weight_matrix)

    def _rescaled(
        self,
        state_vector
        ):
        return (2*state_vector-1) + np.matmul(2*state_vector-1, self.weight_matrix)



def sigmoid(x, l=1, b = 0):
    return 1/(1 + np.exp(-(l*x) + b))

def trivalent(x):
    x[x>0] = 1
    x[x==0] = 0
    x[x<0] = -1
    return x

def bivalent(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x

def tanh(x):
    return np.tanh(x)
















        
