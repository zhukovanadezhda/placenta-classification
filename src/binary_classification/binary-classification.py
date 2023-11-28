# from keras import layers
import pandas as pd
import random

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


if __name__ == "__main__":
    (
        my_train_data,
        my_test_data,
        my_train_labels,
        my_test_labels,
    ) = get_train_test_samples()
