#A novel method for FCM learning with the employment of Neural Networks. 
#Created by Theotziol on 15/9/2023
#Contact info ttziolas@uth.gr


import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
#from sklearn.preprocessing import MinMaxScaler
# from fcm_tensorflow import *
import time
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score



class neural_fcm:
    def __init__(
        self,
        input_shape,
        output_concepts = 1,
        fcm_iter = 1,
        l_slope = 1,
    ):
        '''
        Args:
            input_shape : The shape of each row (tabular data). i.e. for a dataframe with shape (100, 5) input_shape = 5
            output_concepts : The number of concepts to be used as output (tabular dataset). i.e for a timeseries dataset 
        '''
        self.input_shape = input_shape
        self.output_concepts = output_concepts
        self.l_slope = l_slope
        self.fcm_iter = fcm_iter
        self.__nn_model()


    def __nn_model(
        self,
    ):
        inpt = keras.layers.Input(shape = (self.input_shape,)) 
        x1 = keras.layers.Dense(self.input_shape, activation = 'relu')(inpt)
        x1 = keras.layers.Dense(self.input_shape**2, activation = 'relu')(x1) 
        r1 = keras.layers.Reshape((self.input_shape, self.input_shape, 1))(x1) 
        x1 = keras.layers.Conv2D(self.input_shape**2, (3,3), padding = 'same', activation = 'relu')(r1)
        out = keras.layers.Conv2D(1, (1,1), padding = 'same', activation = 'tanh')(x1)
        self.model = keras.Model(inputs = inpt, outputs = out)
        # print(self.model.summary())
    
    def initialize_loss_and_compile(
        self,
        loss,
        regresion_loss_weights = [0.95, 10.0, 1.0],
        classification_loss_weights = [1.5, 1.0, 1.0], # in experiments were  [2.0, 1.0, 1.0]
        lr = 0.001,
    ):
        '''
        Initializes the loss and the metrics
        
            loss : str, the type of loss to be used. Currently only regression loss (MSE) is implemented.
            regression_loss_weights : list, the weights that were defined for the loss. 
                Regression loss has three weights w1, w2, w3. As it is divided in three losses:
                MSE(tn, tn+1)
                w1 : weight of MSE (tn, tn+1) (default = 0.95) the error between the tn concepts activation after fcm \
                    inference and the actual concept values at tn+1
                w2 : weight of MSE (tn_output , tn+1_output) (default 10.0)
                w3 : weight of MSE (current_diagonal - diagonal of zeros) an error to ensure fcm matrix diagonal = 0 and cocnept output = 0
        '''
        self.loss_weights_regression = regresion_loss_weights
        self.classification_loss_weights = classification_loss_weights
        #todo define losses for non_categorical data
        regression_losses = ('regression', 'regresion', 'MSE', 'mse')
        binary_classification_losses = ('bce', 'BCE', 'Binary_Cross_Entropy')
        categorical_losses = ('CCE', 'categorical_cross_entropy', 'cce', 'categorical')
        optimizer = tf.keras.optimizers.Adam(lr) 

        if loss in regression_losses:
            self.loss = self.__custom_loss_regression
            metrics = [] #todo
            
        
        if loss in categorical_losses:
            self.loss = self.__custom_loss_categorical
            metrics = [] #to do

        self.model.compile(optimizer, self.loss)

    def __custom_loss_regression(
        self,
        true,
        predicted
    ):
        '''
        This loss is used for regression. In regression the n_row is given as input and the n+1_row is given as output.
        Therefore, when training a model, both  n_row and n+1_row must be passed as target data to the loss. Here these data are concatenated and the shapes are as following:
            predicted.shape = [batch_size, y, x, 1] with y=x so predicted.shape = [batch_size, x, x, 1]
            true.shape = [batch, x(t0) + x(t1)] or equivalently true.shape = [batch, 2 * x]. 

        '''
        attributes = true.shape[-1]//2
        x = true[:, :attributes]
        y = true[:, attributes:]
        # for i in tf.range(self.fcm_iter):
        #     x = tf.math.add(x, tf.linalg.matmul(x,predicted[:, :, :, 0]))
        #     x = sigmoid(x, self.l_slope)
        #     x = x[0]
        for i in tf.range(self.fcm_iter):
            # x.shape = (batch, features), test.shape = (batch, features, features, channels)
            x = x[:, None, :] + tf.linalg.matmul(x[:, None, :],predicted[:, :, :, 0])
            x = sigmoid(x, self.l_slope)
            x = x[:, 0]
        
        w1 = self.loss_weights_regression[0]
        w2 = self.loss_weights_regression[1]
        w3 = self.loss_weights_regression[2]

        diagonal = tf.linalg.diag_part(predicted[:,:,:,0])
        if x.shape[0] == None:
            shape = (1, x.shape[-1])
        else:
            shape = x.shape
        exp_diagonal = tf.constant(0.0, shape = shape)

        error1 = tf.losses.MSE(x, y)
        error2 = tf.losses.MSE(x[:,-1], y[:, -1] ) #fix -1 to self.output_concept
        error3 = tf.losses.MSE(diagonal, exp_diagonal)
        error4 = tf.losses.MSE(exp_diagonal, predicted[:, -1, :, 0])#the output concept
        error = (w1 * error1) \
                + (w2 * error2) \
                + (w3 * error3) \
                + (w3 * error4)
        return error 
    
    def __custom_loss_categorical(
        self,
        true,
        predicted
        ):
        '''
        This loss is used for classification with one-hot-encoding. In classification the n_row is given as input and the n_labels is given as output.
        Therefore, when training a model, both  n_row and n_labels must be passed as target data to the loss. Here these data are concatenated and the shapes are as following:
            predicted.shape = [batch_size, x, x, 1] and x = input + dummy_label, with label having a dummy value
            true.shape = [batch, x(t0) + y(t0)] or equivalently true.shape = [batch, 2 * x]. 
        '''
        attributes = true.shape[-1] - self.output_concepts 
        x = true[:, :attributes]
        y = true[:, attributes:]
        for i in tf.range(self.fcm_iter):
            # x.shape = (batch, features), test.shape = (batch, features, features, channels)
            x = x[:, None, :] + tf.linalg.matmul(x[:, None, :],predicted[:, :, :, 0])
            x = sigmoid(x, self.l_slope)
            x = x[:, 0]
        
        diagonal = tf.linalg.diag_part(predicted[:,:,:,0])
        outputs = x[:, -self.output_concepts:]
        softoutputs = tf.nn.softmax(outputs)

        cce = tf.keras.losses.CategoricalCrossentropy()
        w1, w2, w3 = self.classification_loss_weights
        error1 = cce(softoutputs, y)
        error2 = tf.math.reduce_mean(tf.math.square(diagonal))
        error3 = tf.math.reduce_mean(tf.math.square(predicted[:, -self.output_concepts:, :, 0]))#the output concept
        error = (w1 * error1) \
                + (w2 * error2) \
                + (w3 * error3) 
        return error
    
    
    def predict_regression(
        self,
        test_data,
        ):

        self.predicted_matrices = self.model.predict(test_data)
        x = test_data
        
        for i in tf.range(self.fcm_iter):
            # x.shape = (batch, features), test.shape = (batch, features, features, channels)
            x = x[:, None, :] + tf.linalg.matmul(x[:, None, :],self.predicted_matrices[:, :, :, 0])
            x = sigmoid(x, self.l_slope)
            x = x[:, 0]
        self.predictions = x[:, -1]
        return x[:, -1]

    def predict_classification(
        self,
        test_data,
        ):

        self.predicted_matrices = self.model.predict(test_data)
        x = test_data
        
        for i in tf.range(self.fcm_iter):
            # x.shape = (batch, features), test.shape = (batch, features, features, channels)
            x = x[:, None, :] + tf.linalg.matmul(x[:, None, :],self.predicted_matrices[:, :, :, 0])
            x = sigmoid(x, self.l_slope)
            x = x[:, 0]
        self.predictions = x[:, -self.output_concepts:]
        return x[:, -self.output_concepts:]
    
    def metrics_classification(
        self,
        y,
        predictions = None,
    ):
        '''
        calculate the classification metrics in a test dataset. Currently works for one hot encoding labels 
        Args:
            y: numpy array of the labels (one hote encoding)
            predictions: numpy array of the predictions. 
                If no array is passed the function utilizes the self.predictions variable,
                calculated with the self.prediction_classification method. 

    F1-Score
        'micro':
        Calculate metrics globally by counting the total true positives, false negatives and false positives.

        'macro':
        Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        '''
        if predictions != None:
            max_indexes_predictions = np.argmax(predictions, axis = -1)
        else:
            max_indexes_predictions = np.argmax(self.predictions, axis = -1)
        
        max_indexes_y = np.argmax(y, axis = -1)
        
        self.accuracy = accuracy_score(max_indexes_y, max_indexes_predictions)
        self.confusion_matrix = confusion_matrix(max_indexes_y, max_indexes_predictions)
        self.f1_score_micro = f1_score(max_indexes_y, max_indexes_predictions, average = 'micro')
        self.f1_score_macro = f1_score(max_indexes_y, max_indexes_predictions, average = 'macro')
        print(f'\nAccuracy = {np.round(self.accuracy,4)},\nF1 (micro) = {np.round(self.f1_score_micro,4)},\nF1 (macro) = {np.round(self.f1_score_macro,4)},\nConfusion Matrix = \n{self.confusion_matrix}\n')

    def statistics_regression_norm(
        self,
        y
    ):
        '''
        Calculate the regression statistics for normalized values
        '''

        self.mse_norm = tf.losses.MSE(self.predictions,y).numpy()
        self.mae_norm = tf.losses.MAE(self.predictions,y).numpy()
        print(f'MSE (norm) = {np.round(self.mse_norm, 4)}\nMAE (norm) = {np.round(self.mae_norm, 4)}\n')

    
    def statistics_regression(
        self,
        real_array_test,
        real_array,
    ):
        '''
        caulculate the regression statistics for the actual scale
        Args:
            real_array_test: the numpy array of the output in actual scale (testing)
            real_array the numpy array in actual scale 
        '''
        assert len(real_array_test) == len(self.predictions)
        min_output =  real_array.min()
        max_output = real_array.max()

        X_std = (self.predictions.numpy() - 0) / (1 - 0) #0 the minimum in the normalized dataset, 1 the maximum in normalized dataset
        self.predictions_actual = X_std * (max_output - min_output) + min_output
        self.mse = tf.losses.MSE(self.predictions_actual,real_array_test).numpy()
        self.mae = tf.losses.MAE(self.predictions_actual,real_array_test).numpy()
        self.mape = tf.losses.MAPE(tf.convert_to_tensor(self.predictions_actual, dtype = tf.float64),tf.convert_to_tensor(real_array_test, dtype=tf.float64)).numpy()
        self.real_array_test = real_array_test
        corr_matrix = np.corrcoef(self.real_array_test, self.predictions_actual)
        corr = corr_matrix[0,1]
        self.R_sq = np.round(corr**2, 3)
        self.m, self.b = np.polyfit(self.real_array_test, self.predictions_actual, 1)

        print(f'MSE = {np.round(self.mse, 4)}\nMAE = {np.round(self.mae, 4)}\nMAPE = {np.round(self.mape, 4)}\n')





class TimeHistory(keras.callbacks.Callback):
    #https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

        
def sigmoid(x,l = 1):
    return 1/(1 + tf.math.exp(-x *l))

def trivalent(x):
    x = x.numpy()
    x[x>0] = 1
    x[x==0] = 0
    x[x<0] = -1
    return tf.convert_to_tensor(x)
    

def bivalent(x):
    x = x.numpy()
    x[x>0] = 1
    x[x<=0] = 0
    return tf.convert_to_tensor(x)
    
def softmax(x):
  return tf.nn.softmax(x)

def tanh(x):
  return tf.keras.activations.tanh(x)


