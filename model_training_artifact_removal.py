import math
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import FastICA

class EarlyStopping:
    def __init__(self, patience, delta, val_type="loss"):
        self.patience = patience
        self.delta = delta
        self.val_type = val_type
        self.counter = 0
        self.best_score = None
        self.best_params = None
        self.early_stop = False

    def __call__(self, val, model):
        if self.val_type == "loss":
            if self.best_score is None:
                self.best_score = val
                self.best_params = model.state_dict()
            elif val > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val
                self.best_params = model.state_dict()
                self.counter = 0
        elif self.val_type == "acc":
            if self.best_score is None:
                self.best_score = val
                self.best_params = model.state_dict()
            elif val < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val
                self.best_params = model.state_dict()
                self.counter = 0


def self_prediction_mask(model, x_train, x_valid, lr=0.01, weight_decay=0, num_epochs=20,
                         batch=64, patience=5, delta=0.01, loss_fn=nn.MSELoss(),
                         shuffle=True, verbose=0):
    """
    Function to train MAEEG model by self-prediction with masking method.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)
    x_train = x_train.to(device)

    if not torch.is_tensor(x_valid):
        x_valid = torch.Tensor(x_valid)
    x_valid = x_valid.to(device)

    model = model.to(device)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = math.ceil(no_samples/batch)
    no_valid_samples = x_valid.shape[0]
    no_valid_loops = math.ceil(no_valid_samples / batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    best_train_loss = float("inf")
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
        for j in range(no_iter_per_epoch):
            model.train()
            out_mask = model.forward_mask(x_train[batch*j:batch*(j+1)])
            x_origin = x_train[batch*j:batch*(j+1)]

            loss = loss_fn(out_mask, x_origin)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if j == no_iter_per_epoch - 1:
                model.eval()
                train_loss_list = []
                for k in range(no_iter_per_epoch):
                    train_out_mask = model.forward_mask(x_train[batch * k: batch * (k + 1)])
                    train_origin = x_train[batch * k: batch * (k + 1)]

                    train_loss = loss_fn(train_out_mask, train_origin)
                    train_loss = train_loss.cpu()
                    train_loss = train_loss.detach().numpy()
                    train_loss_list.append(train_loss)
                ave_train_loss = sum(train_loss_list) / len(train_loss_list)

                # Validation
                valid_loss_list = []
                for k in range(no_valid_loops):
                    valid_out_mask = model.forward_mask(x_valid[batch * k: batch * (k + 1)])
                    valid_origin = x_valid[batch * k: batch * (k + 1)]

                    valid_loss = loss_fn(valid_out_mask, valid_origin)
                    valid_loss = valid_loss.cpu()
                    valid_loss = valid_loss.detach().numpy()
                    valid_loss_list.append(valid_loss)
                ave_valid_loss = sum(valid_loss_list) / len(valid_loss_list)

                early_stopping(val=ave_valid_loss, model=model)

                if ave_valid_loss + delta < best_valid_loss:
                    best_train_loss = ave_train_loss
                    best_valid_loss = ave_valid_loss

                if verbose == 1:
                    print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {ave_train_loss}, valid_loss = {ave_valid_loss}")

        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_params)
            print("Early stopping.")
            break

    return best_train_loss, best_valid_loss





def self_prediction_artifact_removal(model, x_train, x_valid, x_removed_train, x_removed_valid,
                                     lr=0.01, weight_decay=0, num_epochs=20, batch=64, patience=5, delta=0.01,
                                     loss_fn=nn.MSELoss(), shuffle=True, verbose=0):
    """
    Function to train MAEEG model by self-prediction with single artifact removal method.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)
    x_train = x_train.to(device)

    if not torch.is_tensor(x_valid):
        x_valid = torch.Tensor(x_valid)
    x_valid = x_valid.to(device)

    if not torch.is_tensor(x_removed_train):
        x_removed_train = torch.Tensor(x_removed_train)
    x_removed_train = x_removed_train.to(device)

    if not torch.is_tensor(x_removed_valid):
        x_removed_valid = torch.Tensor(x_removed_valid)
    x_removed_valid = x_removed_valid.to(device)

    model = model.to(device)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = math.ceil(no_samples/batch)
    no_valid_samples = x_valid.shape[0]
    no_valid_loops = math.ceil(no_valid_samples / batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    best_train_loss = float("inf")
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
            x_removed_train = x_removed_train[random_index]
        for j in range(no_iter_per_epoch):
            model.train()
            out_filter = model(x_train[batch*j:batch*(j+1)])
            x_filter = x_removed_train[batch*j:batch*(j+1)]

            loss = loss_fn(out_filter, x_filter)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if j == no_iter_per_epoch - 1:
                model.eval()
                train_loss_list = []
                for k in range(no_iter_per_epoch):
                    train_out_filter = model(x_train[batch * k: batch * (k + 1)])
                    train_filter = x_removed_train[batch * k: batch * (k + 1)]

                    train_loss = loss_fn(train_out_filter, train_filter)
                    train_loss = train_loss.cpu()
                    train_loss = train_loss.detach().numpy()
                    train_loss_list.append(train_loss)
                ave_train_loss = sum(train_loss_list) / len(train_loss_list)

                # Validation
                valid_loss_list = []
                for k in range(no_valid_loops):
                    valid_out_filter = model(x_valid[batch * k: batch * (k + 1)])
                    valid_filter = x_removed_valid[batch * k: batch * (k + 1)]

                    valid_loss = loss_fn(valid_out_filter, valid_filter)
                    valid_loss = valid_loss.cpu()
                    valid_loss = valid_loss.detach().numpy()
                    valid_loss_list.append(valid_loss)
                ave_valid_loss = sum(valid_loss_list) / len(valid_loss_list)

                early_stopping(val=ave_valid_loss, model=model)

                if ave_valid_loss + delta < best_valid_loss:
                    best_train_loss = ave_train_loss
                    best_valid_loss = ave_valid_loss

                if verbose == 1:
                    print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {ave_train_loss}, valid_loss = {ave_valid_loss}")

        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_params)
            print("Early stopping.")
            break

    return best_train_loss, best_valid_loss



def self_prediction_multiple_artifact_removal(model, x_train, x_valid, x_removed_train, x_removed_valid,
                                              lr=0.01, weight_decay=0, num_epochs=20, batch=64, patience=5, delta=0.01,
                                              loss_fn=nn.MSELoss(), shuffle=True, verbose=0):
    """
    Function to train MAEEG model by self-prediction with multiple artifact removal method.
    """

    assert isinstance(x_removed_train, (list, tuple)) and isinstance(x_removed_valid, (list, tuple)), \
        ("x_removed_train and x_removed_valid should be list or tuple.")

    assert len(x_removed_train) == len(x_removed_valid), \
        ("x_removed_train and x_removed_valid should have the same size.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)
    x_train = x_train.to(device)

    if not torch.is_tensor(x_valid):
        x_valid = torch.Tensor(x_valid)
    x_valid = x_valid.to(device)

    for i in range(len(x_removed_train)):
        if not torch.is_tensor(x_removed_train[i]):
            x_removed_train[i] = torch.Tensor(x_removed_train[i])
        x_removed_train[i] = x_removed_train[i].to(device)

        if not torch.is_tensor(x_removed_valid[i]):
            x_removed_valid[i] = torch.Tensor(x_removed_valid[i])
        x_removed_valid[i] = x_removed_valid[i].to(device)

    model = model.to(device)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = math.ceil(no_samples/batch)
    no_valid_samples = x_valid.shape[0]
    no_valid_loops = math.ceil(no_valid_samples / batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    best_train_loss = float("inf")
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
            x_removed_train = [x[random_index] for x in x_removed_train]
        for j in range(no_iter_per_epoch):
            model.train()
            out_filter = model(x_train[batch*j:batch*(j+1)])

            loss = 0
            for k in range(len(x_removed_train)):
                x_filter = x_removed_train[k][batch*j:batch*(j+1)]
                loss = loss + loss_fn(out_filter, x_filter)

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            if j == no_iter_per_epoch - 1:
                model.eval()
                train_loss_list = []
                for k in range(no_iter_per_epoch):
                    train_out_filter = model(x_train[batch * k: batch * (k + 1)])

                    train_loss = 0
                    for m in range(len(x_removed_train)):
                        train_filter = x_removed_train[m][batch * k: batch * (k + 1)]
                        train_loss = train_loss + loss_fn(train_out_filter, train_filter)
                    train_loss = train_loss.cpu()
                    train_loss = train_loss.detach().numpy()
                    train_loss_list.append(train_loss)
                ave_train_loss = sum(train_loss_list) / len(train_loss_list)

                # Validation
                valid_loss_list = []
                for k in range(no_valid_loops):
                    valid_out_filter = model(x_valid[batch * k: batch * (k + 1)])

                    valid_loss = 0
                    for m in range(len(x_removed_valid)):
                        valid_filter = x_removed_valid[m][batch * k: batch * (k + 1)]
                        valid_loss = valid_loss + loss_fn(valid_out_filter, valid_filter)
                    valid_loss = valid_loss.cpu()
                    valid_loss = valid_loss.detach().numpy()
                    valid_loss_list.append(valid_loss)
                ave_valid_loss = sum(valid_loss_list) / len(valid_loss_list)

                early_stopping(val=ave_valid_loss, model=model)

                if ave_valid_loss + delta < best_valid_loss:
                    best_train_loss = ave_train_loss
                    best_valid_loss = ave_valid_loss

                if verbose == 1:
                    print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {ave_train_loss}, valid_loss = {ave_valid_loss}")

        if early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_params)
            print("Early stopping.")
            break

    return best_train_loss, best_valid_loss



def classifier_train(model, x_train, y_train, lr=0.01, num_epochs=20, batch=64,
                     loss_fn=nn.BCELoss(), shuffle=True, verbose=0):
    """
    Function to train MAEEG classifier.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)
    x_train = x_train.to(device)

    if not torch.is_tensor(y_train):
        y_train = torch.Tensor(y_train)
    y_train = y_train.to(device)

    model = model.to(device)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = math.ceil(no_samples/batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss_list = []
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
            y_train = y_train[random_index]
        for j in range(no_iter_per_epoch):
            if j != no_iter_per_epoch - 1:
                out = model(x_train[batch*j:batch*(j+1)])
                loss = loss_fn(out, y_train[batch*j:batch*(j+1)])
            else:
                out = model(x_train[batch*j:])
                loss = loss_fn(out, y_train[batch*j:])

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            loss = loss.cpu()
            loss = loss.detach().numpy()
            train_loss_list.append(loss)

            if verbose == 1 and j == no_iter_per_epoch - 1:
                train_acc_list = []
                for i in range(no_iter_per_epoch):
                    out = model(x_train[batch*i: batch*(i+1)])
                    out[out>0.5] = 1
                    out[out<=0.5] = 0
                    acc = (out == y_train[batch*i: batch*(i+1)]).sum() / batch
                    acc = acc.cpu()
                    acc = acc.detach().numpy()
                    train_acc_list.append(acc)
                acc = sum(train_acc_list)/len(train_acc_list)
                print(f"Epoch {epoch+1}/{num_epochs}: loss = {sum(train_loss_list)/len(train_loss_list): 4f}, "
                      f"accuracy = {acc: 4f}")



def classifier_train_valid(model, x_train, y_train, x_valid, y_valid, lr=0.01, weight_decay=1e-4,
                           num_epochs=20, batch=64, loss_fn=nn.BCELoss(), use_early_stopping=True,
                           patience=5, delta=0.01, val_type="acc", shuffle=True, verbose=0):
    """
    Function to train MAEEG classifier with validation.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(x_train):
        x_train = torch.Tensor(x_train)
    x_train = x_train.to(device)

    if not torch.is_tensor(y_train):
        y_train = torch.Tensor(y_train)
    y_train = y_train.to(device)

    if not torch.is_tensor(x_valid):
        x_valid = torch.Tensor(x_valid)
    x_valid = x_valid.to(device)

    if not torch.is_tensor(y_valid):
        y_valid = torch.Tensor(y_valid)
    y_valid = y_valid.to(device)

    model = model.to(device)

    no_samples = x_train.shape[0]
    no_iter_per_epoch = math.ceil(no_samples/batch)
    no_valid_samples = x_valid.shape[0]
    no_valid_loops = math.ceil(no_valid_samples / batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, delta=delta, val_type=val_type)

    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    best_train_acc = 0.0
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        if shuffle:
            random_index = torch.randperm(no_samples)
            x_train = x_train[random_index]
            y_train = y_train[random_index]
        for j in range(no_iter_per_epoch):
            model.train()
            out = model(x_train[batch*j:batch*(j+1)])
            loss = loss_fn(out, y_train[batch*j:batch*(j+1)])

            # Calculate gradient
            loss.backward()

            # Update weights
            optimizer.step()

            # Reset gradient
            optimizer.zero_grad()

            model.eval()
            if j == no_iter_per_epoch - 1:
                model.eval()
                train_loss_list = []
                train_acc_list = []
                for k in range(no_iter_per_epoch):
                    train_out = model(x_train[batch * k:batch * (k + 1)])
                    train_loss = loss_fn(train_out, y_train[batch * k:batch * (k + 1)])
                    train_out[train_out > 0.5] = 1
                    train_out[train_out <= 0.5] = 0
                    train_acc = (train_out == y_train[batch * k:batch * (k + 1)]).sum() / train_out.shape[0]

                    train_loss = train_loss.cpu()
                    train_loss = train_loss.detach().numpy()
                    train_loss_list.append(train_loss)
                    train_acc = train_acc.cpu()
                    train_acc = train_acc.detach().numpy()
                    train_acc_list.append(train_acc)

                ave_train_loss = sum(train_loss_list) / len(train_loss_list)
                ave_train_acc = sum(train_acc_list) / len(train_acc_list)

                # Validation
                valid_loss_list = []
                valid_acc_list = []
                for k in range(no_valid_loops):
                    valid_out = model(x_valid[batch*k:batch*(k + 1)])
                    valid_loss = loss_fn(valid_out, y_valid[batch*k:batch*(k + 1)])
                    valid_out[valid_out > 0.5] = 1
                    valid_out[valid_out <= 0.5] = 0
                    valid_acc = (valid_out == y_valid[batch*k:batch*(k+1)]).sum() / valid_out.shape[0]

                    valid_loss = valid_loss.cpu()
                    valid_loss = valid_loss.detach().numpy()
                    valid_loss_list.append(valid_loss)
                    valid_acc = valid_acc.cpu()
                    valid_acc = valid_acc.detach().numpy()
                    valid_acc_list.append(valid_acc)

                ave_valid_loss = sum(valid_loss_list) / len(valid_loss_list)
                ave_valid_acc = sum(valid_acc_list) / len(valid_acc_list)

                if use_early_stopping:
                    if val_type == "loss":
                        early_stopping(val=ave_valid_loss, model=model)
                    elif val_type == "acc":
                        early_stopping(val=ave_valid_acc, model=model)

                if val_type == "loss" and (ave_valid_loss + delta < best_valid_loss):
                    best_train_loss = ave_train_loss
                    best_train_acc = ave_train_acc
                    best_valid_loss = ave_valid_loss
                    best_valid_acc = ave_valid_acc
                elif val_type == "acc" and (ave_valid_acc + delta > best_valid_acc):
                    best_train_loss = ave_train_loss
                    best_train_acc = ave_train_acc
                    best_valid_loss = ave_valid_loss
                    best_valid_acc = ave_valid_acc


                if verbose == 1:
                    print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {ave_train_loss: 4f}, "
                          f"train_acc = {ave_train_acc: 4f}, valid_loss = {ave_valid_loss: 4f}, "
                          f"valid_acc = {ave_valid_acc: 4f}")

        if use_early_stopping and early_stopping.early_stop:
            model.load_state_dict(early_stopping.best_params)
            print("Early stopping.")
            break

    return best_train_loss, best_train_acc, best_valid_loss, best_valid_acc


def remove_frequency_component(x, freq_min, freq_max):
    """
    Function to remove frequencies out of the range.
    """

    if torch.is_tensor(x):
        x = x.cpu()
    x_fft = np.fft.rfft(x, axis=-1)

    x_fft[:, :, :freq_min] = 0
    x_fft[:, :, freq_max + 1:] = 0
    x_ifft = np.fft.irfft(x_fft)
    return x_ifft



def ica_and_frequency_removal(x, freq_min, freq_max, n_components=None, algorithm='parallel',
                              whiten='unit-variance', fun='logcosh', fun_args=None, max_iter=200,
                              tol=0.0001, w_init=None, whiten_solver='svd', random_state=None):
    """
    Function to apply ICA and frequency removal to EEG.
    """

    if torch.is_tensor(x):
        x = x.cpu()
        x = x.detach().numpy()
    batch, num_channel, time_len = x.shape
    x_new = x.transpose(0, 2, 1)
    x_new = x_new.reshape((-1, num_channel))
    ica = FastICA(n_components=n_components, algorithm=algorithm, whiten=whiten, fun=fun, fun_args=fun_args,
                  max_iter=max_iter, tol=tol, w_init=w_init, random_state=random_state)

    x_ica = ica.fit_transform(x_new)
    x_ica = x_ica.reshape((batch, time_len, num_channel)).transpose(0, 2, 1)

    x_fft = np.fft.rfft(x_ica, axis=-1)

    x_fft[:, :, :freq_min] = 0
    x_fft[:, :, freq_max + 1:] = 0
    x_ifft = np.fft.irfft(x_fft)
    x_ifft = x_ifft.transpose(0, 2, 1)
    x_ifft = x_ifft.reshape((-1, num_channel))

    x_inverse = ica.inverse_transform(x_ifft)
    x_inverse = x_inverse.reshape((batch, time_len, num_channel))
    x_inverse = x_inverse.transpose(0, 2, 1)
    return x_inverse


def standardize_eeg(x, scaler, is_train=False):
    """
    Function to standardize EEG.
    """
    num_windows, num_channels, t_steps = x.shape
    if is_train:
        x_copy = x.copy()
        scaler_fit_eeg(x_copy, scaler)
    x = np.transpose(x, (0, 2, 1))
    x = x.reshape((-1, num_channels))
    x = scaler.transform(x)

    x = x.reshape((num_windows, t_steps, num_channels))
    x = np.transpose(x, (0, 2, 1))
    return x

def scaler_fit_eeg(x, scaler):
    """
    Function to let scaler fit EEG.
    """
    num_windows, num_channels, t_steps = x.shape
    x = np.transpose(x, (0, 2, 1))
    x = x.reshape((-1, num_channels))
    scaler.fit(x)
