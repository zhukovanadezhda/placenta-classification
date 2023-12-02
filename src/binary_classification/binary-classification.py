# from keras import layers
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import random
import matplotlib.pyplot as plt

# CONSTANTS
DATA_FILE = "data/data_pred_ci2.csv"
DATA = pd.read_csv(DATA_FILE, index_col=0)
SEED = 42


# SCRIPT
def get_train_test_samples():
    nb_molecules = DATA.shape[0]
    nb_train = int((2 / 3) * nb_molecules)
    random.seed(SEED)
    total_names = set(DATA.index)
    train_names = set(random.sample(total_names, nb_train))
    test_names = total_names.difference(train_names)

    # VERIFICATION
    # print(len(total_names) == (len(train_names) + len(test_names)))
    # print(train_names.intersection(test_names))

    train_labels = DATA.loc[list(train_names), ["CI2"]]
    test_labels = DATA.loc[list(test_names), ["CI2"]]
    train_data = DATA.loc[list(train_names)]
    test_data = DATA.loc[list(test_names)]
    del train_data["CI2"]
    del test_data["CI2"]
    return (train_data, test_data, train_labels, test_labels)


def display_plot(res_train, param):
  plt.plot(res_train.history[param], label = 'Train ' + param)
  plt.title(param + ' over epochs')
  plt.xlabel('epochs')
  plt.ylabel(param)
  plt.legend()
  plt.savefig(param + '.png')


if __name__ == "__main__":
    (
        my_train_data,
        my_test_data,
        my_train_labels,
        my_test_labels,
    ) = get_train_test_samples()

    #print(my_train_data.shape)
    #print(my_test_data.shape)
    #print(my_train_labels)
    print(DATA)

    #Construction du reseau
    model = Sequential() #creation reseau vide
    model.add(Dense(16, activation = "relu", input_shape = (DATA.shape[1]-1,))) #ajout d'une premiere couche dense d'entrée | input_shape ???????
    model.add(Dense(16, activation = "relu")) #couche cachee
    model.add(Dense(1, activation = "sigmoid")) #couche de sortie

    #Compiler le model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])

    #Entrainer le modele
    res_train = model.fit(my_train_data, my_train_labels, batch_size=32, epochs=10)

    display_plot(res_train, 'accuracy')
    display_plot(res_train, 'loss')

    # loss_and_metrics = model.evaluate(my_test_data, my_test_labels)
    # print(loss_and_metrics)
    # print('Loss = ',loss_and_metrics[0])
    # print('Accuracy = ',loss_and_metrics[1])

