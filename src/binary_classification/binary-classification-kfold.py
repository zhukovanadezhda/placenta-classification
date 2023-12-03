# from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

# CONSTANTS
DATA_FILE = "data/data_pred_ci2.csv"
DATA = pd.read_csv(DATA_FILE, index_col=0)
SEED = 42
num_folds = 3


def display_plot(histories, param):
    plt.figure()
    for no_fold, res_train in enumerate(histories) : 
        legend_no_fold = str(no_fold+1) + " fold"
        plt.plot(res_train.history[param], label = legend_no_fold)
    plt.title(param + ' over epochs')
    plt.xlabel('epochs')
    plt.ylabel(param)
    plt.legend()
    plt.savefig(param + '.png')


if __name__ == "__main__":

    inputs = DATA.copy(deep=True)
    del inputs["CI2"]
    targets = DATA.loc[:,["CI2"]]

    # contenants des performances par fold
    acc_per_fold = []
    loss_per_fold = []
    kfold = KFold(n_splits=num_folds, shuffle=True)

    #contenant les history de chaque fold
    history_per_fold = []
    
    fold_no = 1
    for train, test in kfold.split(inputs, targets):

        #Construction du reseau
        model = Sequential() #creation reseau vide
        model.add(Dense(16, activation = "relu", input_shape = (DATA.shape[1]-1,))) #ajout d'une premiere couche dense d'entrée | input_shape ???????
        model.add(Dense(16, activation = "relu")) #couche cachee
        model.add(Dense(1, activation = "sigmoid")) #couche de sortie

        #Compiler le model
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        #Entrainer le modele
        res_train = model.fit(inputs.iloc[train], targets.iloc[train], validation_split=0.33, batch_size=32, epochs=10)
        history_per_fold.append(res_train)

        #La fonction evaluate verifie la performance du model sur les donnees de test
        print("------------------------------Evaluation du model------------------------------")

        scores = model.evaluate(inputs.iloc[test], targets.iloc[test])
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    
    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    display_plot(history_per_fold, "accuracy")
    display_plot(history_per_fold, "loss")
    display_plot(history_per_fold, "val_accuracy")
    display_plot(history_per_fold, "val_loss")