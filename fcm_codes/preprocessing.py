# Created by Theotziol 21/3/2024
# Contains preprocessing methods applied in pandas dataframes
import pandas as pd 
import numpy as np 

def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())

    return df_norm

def split_df(df_normalized, ratio = 0.8, numpy = True):
    '''
    Splits the dataframe into training/testing datasets
    Args:
        df_normalized : a pandas dataframe. Suggested to be normalized prior to splitting
        ratio : float. The ratio that will be kept for training (Default = 0.8, or 80%).
        numpy: Boolean. Whether to return the training/testing datasets as a numpy array (Default = True).
    Returns:
        training: pd dataframe if numpy = False, numpy array otherwise
        testing: pd dataframe if numpy = False, numpy array otherwise
    '''
    index = int(ratio * len(df_normalized))
    training = df_normalized.iloc[:index]
    testing = df_normalized.iloc[index:]
    if numpy:
      return training.to_numpy(), testing.to_numpy()
    else:
      return training,testing


def split_labels(df, labels_index = -1, value = 0.5, shuffle = True):
    '''
    function for classification tasks. 
    It splits the original dataframe into 1) input_df and 2) labels_df. 
    FCM classification requires the input vector to have the shape of inputs+outputs.
    Thus, this function separates the original label values and replaces them with the dummy {value}.
    Args:
        df: the pandas dataframe
        labels_index : int. the column which the separation will began. (default = -1)
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
    df_labels = input_df[df.columns[labels_index:]]
    input_df[df.columns[labels_index:]] = value
    return input_df, df_labels

def split_train_test(x_array, y_array, ratio = 0.8):
    '''
    returns:
        x_train, x_test, y_train, y_test arrays
    '''
    x_train = x_array[:int(ratio*len(x_array))]
    x_test = x_array[int(ratio*len(x_array)):]
    y_train = y_array[:int(ratio*len(y_array))]
    y_test = y_array[int(ratio*len(y_array)):]
    return x_train, x_test, y_train, y_test



def split_input_target(df, timesteps = 1, fcm_concat = True):
    '''
    Regression function for Neural-FCM
    Recieves a df and splits it into input output
    Args:
        df : type pd.DataFrame
        timesteps: int, the time distance to predict
        fcm_concat: boolean, whether to concat x and y in order to be used with the neural fcm
    Returns:
        x, y numpy arrays
    '''
    y = df.iloc[timesteps:].to_numpy()
    x = df.iloc[:-timesteps].to_numpy()
    if fcm_concat:
        y = np.concatenate([x,y], axis = -1)
    return x,y


###new preprocessing function

def convert_to_categorical(df, column, shuffle = True):
    '''
    converts a pandas dataframe output column into categorical columns (1 column per class label)
    use the shuffle boolean to randomly re-arrange the rows
    '''
    df = df.copy()
    labels = np.unique(df[column])
    zeros = np.zeros(len(df))
    for i in range(len(labels)):
        df[labels[i]] = zeros
        df[labels[i]][df[column] == labels[i]] = 1
    df.pop(column)
    if shuffle:
        df = df.sample(frac = 1).reset_index(drop=True)
    return df


def text_to_numerical(df, column):
    '''
    converts a pandas column that contains text (dtype == object) into integers {0, ...,  n} 

    '''
    df = df.copy()
    uniqs = np.unique(df[column])
    numbers = [i for i in range(len(uniqs))]
    for i,value in enumerate(uniqs):
        df[column][df[column] == value] = numbers[i]
    df[column] = df[column].astype('int16')
    return df



