import numpy as np
import pandas as pd


class subDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target):
        super(subDataset, self).__init__()
        self.X_input = X_input
        self.X_target = X_target
    def __len__(self):
        return (self.X_input).shape[0]
    def __getitem__(self, idx):
        return (self.X_input[idx], self.X_target[idx])


def getGenerator(data_name):
    """
    Arguments:
        data_name    - (string) name of data file
    Returns:
        DataGenerator class that fits data_name
    """

    return GeneralGenerator


class DataGenerator(object):
    def __init__(self, data, target_col, indep_col, win_size,pre_T, train_share=0.9, is_stateful=False, normalize_pattern=2):
        self.data = data
        self.data_x = data[:, indep_col[0]:indep_col[1]]
        self.train_share = train_share
        self.win_size = win_size
        self.pre_T = pre_T
        self.target_col = target_col
        self.indep_col = indep_col
        self.normalize_pattern = normalize_pattern
        self.is_stateful = is_stateful

    def normalize(self, X):
        if self.normalize_pattern == 0:
            pass
        elif self.normalize_pattern == 1:
            maximums = np.max(np.abs(X), axis=0)
            X = X / maximums
        elif self.normalize_pattern == 2:
            means = np.mean(X, axis=0, dtype=np.float32)
            stds = np.std(X, axis=0, dtype=np.float32)
            X = (X - means) / (stds + (stds == 0) * .001)
        else:
            raise Exception('invalid normalize_pattern')
        return X.astype(np.float32), means, stds

    def with_target(self):
        dta_target = self.data[:, self.target_col]
        dta_x_norm, _, _ = self.normalize(self.data_x)
        #norm_y, y_mean, y_std = self.normalize(dta_target)
        n = len(self.data) - self.pre_T
        if n < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_all = []
        Y_all = []

        if self.is_stateful:
            for i in range(self.win_size, n, self.win_size):
                tmx = dta_x_norm[i - self.win_size:i]

                X_all.append(tmx)
                Y_all.append(dta_target[i])
        else:
            for i in range(self.win_size, n):
                tmx = dta_x_norm[i - self.win_size:i]
                X_all.append(tmx)
                #tmy = dta_target[i - self.win_size:i + self.pre_T]
                tmy = dta_target[i:i + self.pre_T]
                # tmy = np.expand_dims(list_target[i:i+5], axis=1)
                Y_all.append(tmy)

        X_all = np.array(X_all)
        Y_all = np.array(Y_all)
        row_num = X_all.shape[0]
        n_train = int(row_num * self.train_share)
        # n_valid = int(self.row_num * (train_share[0] + train_share[1]))
        n_test = row_num
        X_train = X_all[:n_train]
        # X_valid = X_all[n_train:n_valid]
        X_test = X_all[n_train:n_test]
        Y_train = Y_all[:n_train]
        Y_test = Y_all[n_train:n_test]
        norm_y, y_mean, y_std = self.normalize(Y_train)

        return X_train, X_test, Y_train, Y_test, y_mean, y_std


class GeneralGenerator(DataGenerator):
    def __init__(self, data_path, target_col, indep_col, win_size, pre_T, train_share=0.9, is_stateful=False, normalize_pattern=2):
        X = pd.read_csv(data_path, dtype=np.float32)
        super(GeneralGenerator, self).__init__(data=X.values,
                                               target_col=target_col,
                                               indep_col=indep_col,
                                               win_size=win_size,
                                               pre_T=pre_T,
                                               train_share=train_share,
                                               is_stateful=is_stateful,
                                               normalize_pattern=normalize_pattern
                                               )
