"""
This Python script takes train features and labels, test features and labels, output path, epoch time, learning rate as input
outputs train error, test error and f1 score in a csv file
To automatelly run batch cross vaildation, The script was wrapped in a bash script 'cv.sh'
Don't have to run this script manually
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy_loss(y_true,y_pred):
    m = y_true.shape[0]
    p = np.clip(y_pred, 1e-10, 1-1e-10)
    log_likelihood = -np.log(p[np.arange(m), y_true])
    loss = np.sum(log_likelihood) / m
    
    return loss

def train_valid(
    #theta : np.ndarray, # shape (D, K) where D is feature dim, K is number of classes
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    X_val : np.ndarray,
    y_val : np.ndarray,
    num_epoch : int, 
    learning_rate : float
) -> None:
    # TODO: Implement `train` using vectorization
    # Extand the theta, add an intercept term. Add it to the first column of X
    K = len(np.unique(y))
    N = len(y)
    
    X = np.insert(X, 0, 1, axis=1)
    X_val = np.insert(X_val, 0, 1, axis=1)
    theta = np.zeros((X.shape[1], K))

    # Create a array to store the negative log likelihood for each epoch
    loss = np.zeros(num_epoch)
    loss_val = np.zeros(num_epoch)

    for epoch in range(num_epoch):
        logits = np.dot(X, theta)
        y_prob = softmax(logits)
        
        # calculate gradient
        y_one_hot = np.eye(K)[y]
        gradient = np.dot(X.T, (y_prob - y_one_hot)) / N
        
        # update weight martix
        theta -= learning_rate * gradient
        
        # calculate loss
        loss[epoch] = cross_entropy_loss(y, y_prob)
        logits_val = np.dot(X_val, theta)
        y_prob_val = softmax(logits_val)
        loss_val[epoch] = cross_entropy_loss(y_val, y_prob_val)
    return theta, loss

def predict(
    theta : np.ndarray,
    X : np.ndarray,
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    X = np.insert(X, 0, 1, axis=1)
    # create a array for y_predict
    y_pred = np.zeros(len(X))
    # Calculate the probability of each class
    prob = softmax(np.dot(X, theta))
    # Find the class with the highest probability
    y_pred = np.argmax(prob, axis=1)

    return y_pred

        
def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    error = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y[i]:
            error += 1
    return error/len(y_pred)

# Write output for labels
def write_labels(
    y_pred:np.ndarray,
    dicts:dict,
    file:str
) -> None:
    # Convert the integer labels back to the original string labels
    int_to_label = {idx: label for label, idx in dicts.items()}
    y_pred_label = np.array([int_to_label[label] for label in y_pred])
    with open(file,'w') as f:
        for i in range(len(y_pred_label)):
            f.write(str(y_pred_label[i])+'\n')

# Load tsv
def load_tsv_dataset(file):
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8')
    return dataset

# Load the csv data
def load_data(file):
    # csv ,first row is the feature names, the rest are the data
    dataset = np.loadtxt(file, delimiter=',', comments=None, encoding='utf-8', skiprows=1) 
    return dataset

# load the labels
def load_labels(file):
    # csv, only one column, the labels, one header, string!
    labels = np.loadtxt(file, delimiter=',', comments=None,  dtype='str', skiprows=1)
    # create a map for string to int
    label_to_int = {label: idx for idx, label in enumerate(np.unique(labels))}
    # turn the label to int
    encoded_labels = np.array([label_to_int[label] for label in labels])
    return encoded_labels, label_to_int
    

# Plot the train and validation loss for each epoch
def plot_loss(loss, loss_val, filename):
    plt.figuresize=(10,5)
    plt.plot(loss, label='train')
    plt.plot(loss_val, label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('The average negative log-likelihood over Epoch')
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("train_label", type=str, help='path to training labels')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("validation_label", type=str, help='path to validation labels')
    #parser.add_argument("test_input", type=str, help='path to formatted test data')
    #parser.add_argument("test_label", type=str, help='path to test labels')
    #parser.add_argument("train_out", type=str, help='file to write train predictions to')
    #parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    # Read in feature engineered data
    X_train = load_data(args.train_input)
    y_train,train_dict = load_labels(args.train_label)
    y_valid,vaild_dict = load_labels(args.validation_label)
    X_valid = load_data(args.validation_input)
    #y_test,test_dict =  load_labels(args.test_label)
    #X_test =  load_data(args.test_input)

    # Use SGD to train the model
    
    theta,loss = train_valid(X_train, y_train, X_valid, y_valid, args.num_epoch, args.learning_rate)

    # Predict the labels
    y_pred_train = predict(theta, X_train)
    #y_pred_test = predict(theta,X_test)
    y_pred_test = predict(theta,X_valid) #cross validation

    # Write the error rates
    with open(args.metrics_out, 'w') as f:
        f.write('error(train), ' + str(format(compute_error(y_pred_train, y_train),'6f')) + '\n')
        f.write('error(test),' + str(format(compute_error(y_pred_test,y_valid),'6f')) + '\n')
        f.write('f1_score(train),' + str(format(f1_score(y_train, y_pred_train, average='weighted'),'6f')) + '\n')
    
    # Write the labels
    #write_labels(y_pred_train,train_dict,args.train_out)
    #write_labels(y_pred_test,test_dict,args.test_out)

"""
    # Plot the loss
    plot_loss(loss_train, loss_val, 'loss.png')
"""


    



