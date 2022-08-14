from files.utils.constants import *
from files.pif.pif import *
from files.utils.utility_functions import *
from files.utils.dataset_creator import *
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.utils.data as Data
import time
import sys
from datetime import timedelta
from files.utils.utility_functions import euclidean_distance


def fast_train(epochs, model, dataset):
    for t in range(epochs):
        for point in dataset:
            point = point.reshape(1, -1)
            model.optimizer.zero_grad()
            prediction = model(point)
            loss = model.loss(prediction, point)
            loss.backward()
            model.optimizer.step()


def train(epochs, model, dataset, print_training=True):
    # model.train(True)
    mean_loss_epochs = 0
    chars_to_print = 30
    loss_list_epochs = []

    for t in range(1, epochs+1):
        start_time = time.monotonic()
        loss_list = []

        for i, (b_x, b_y) in enumerate(dataset):
            if b_x is None or b_y is None:
                raise Exception()
            model.optimizer.zero_grad()   # clear gradients for next train
            # input x and predict based on x
            prediction = model(b_x)

            if b_y.shape != prediction.shape:
                b_y = b_y.reshape(prediction.shape)

            # must be (1. nn output, 2. target)
            loss = model.loss(prediction, b_y)
            loss.backward()                     # backpropagation, compute gradients
            model.optimizer.step()              # apply gradients
            loss_list.append(loss.item())
        loss_list_epochs.append(np.mean(loss_list))

        if print_training:
            mean_loss_epochs = np.mean(loss_list_epochs)
            perc = int(t/epochs * chars_to_print)
            dt = timedelta(seconds=time.monotonic()-start_time)
            sys.stdout.write(f"\rEpoch {t}/{epochs}: [" + "="*perc + ">" + "."*(
                chars_to_print-perc) + f"] ({int(t/epochs*100)}%) ETA: {dt} Mean Loss: {mean_loss_epochs:.4f}")
            sys.stdout.flush()

    if print_training:
        dt = timedelta(seconds=time.monotonic()-start_time)
        print(f"\rEpoch {epochs}/{epochs}: [" + "="*chars_to_print +
              f"] (100%) ETA: {dt} Mean Loss: {mean_loss_epochs:.4f}")

    return model, loss_list_epochs


class NeuralNetwork(torch.nn.Module):

    def __init__(self, loss=torch.nn.MSELoss(), neurons=None, layers=None, activation=torch.tanh, device="cpu"):
        assert (neurons is None and layers is not None) or (
            neurons is not None and layers is None)
        super(NeuralNetwork, self).__init__()

        if layers is None:
            self.neurons = neurons
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(neuron, self.neurons[i+1]) for i, neuron in enumerate(self.neurons[:-1])])
        else:
            if isinstance(layers, torch.nn.ModuleList):
                self.layers = layers
            elif isinstance(layers, list):
                self.layers = torch.nn.ModuleList(layers)
            self.neurons = [l.in_features for l in self.layers] + \
                [self.layers[-1].out_feaures]

        self.loss = loss
        self.activation = activation
        self.device = torch.device(device) if device.lower(
        ) == "cuda" and torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

    def tensor_from_np(self, arrs):
        res = [torch.FloatTensor(arr).to(self.device) if arr is not None else None if not isinstance(
            arr, torch.Tensor) else arr for arr in arrs]
        res = [arr.unsqueeze(1) if arr is not None and len(
            arr.shape) == 1 else arr for arr in res]
        if len(res) == 1:
            return res[0]
        else:
            return res

    def forward(self, x):
        x = self.tensor_from_np([x])
        # if len(x.shape) < 2: x = x.reshape(1,-1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            # x = torch.tanh(layer(x))
        yhat_batch = self.layers[-1](x)
        return yhat_batch

    def predict(self, X, return_class=False):
        x = self.tensor_from_np([X])
        yhat_batch = self(x)
        if return_class:
            return torch.LongTensor([1 if yhat > 0.5 else 0 for yhat in yhat_batch]).numpy()
        else:
            return yhat_batch.detach().numpy()

    def predict_proba(self, x):
        x = self.tensor_from_np([x])
        probs = self.predict(x, return_class=False)
        return np.dstack((1-probs, probs))[:, 0, :]

    def score(self, X, y):
        X, y = self.tensor_from_np([X, y])
        with torch.no_grad():
            preds = self(X)
            y = y.reshape(preds.shape)
            l = self.loss(preds, y)
        return l.detach().numpy().mean()

    def get_params(self, deep=False):
        return {
            'neurons': self.neurons,
            'layers': list(self.layers),
            'loss': self.loss,
            'activation': self.activation,
            'device': self.device,
        }

    def fit(self, data, epochs=5, lr=1e-2, bs=16, print_training=False):
        data = self.tensor_from_np([data])

        torch_dataset = Data.TensorDataset(data, data)  # [:,1].unsqueeze(1))
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=bs,
            # num_workers=16,
            shuffle=True)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        return train(epochs=epochs,
                     model=self,
                     dataset=loader,
                     print_training=print_training
                     )

    def get_residuals(self, data):
        preds = self.predict(data)
        # return np.array([euclidean_distance(p, pred)
        #                      for p, pred in zip(data, preds)])
        return np.linalg.norm(data - preds, axis=1)

    def get_inliers(self, data, in_th=1, v=None):
        residuals = self.get_residuals(data)
        return data[np.where(residuals < in_th)]


class AEModel(NeuralNetwork):

    def __init__(self, loss=torch.nn.MSELoss(), n_inputs=2, n_first_hidden=0, n_outputs=2, step=2, activation=torch.tanh):
        self.n_inputs = n_inputs
        self.n_first_hidden = n_first_hidden
        self.n_outputs = n_outputs
        neurons = [n_inputs, n_first_hidden]
        if n_first_hidden == 0:
            neurons = [n_inputs, 1, n_outputs]
        else:
            n = n_first_hidden
            while max(1, n) > 1:
                n = int(n/step)
                n = max(1, n)
                neurons.append(n)

            while min(n, n_first_hidden) < n_first_hidden:
                n *= step
                neurons.append(n)

            neurons.append(n_outputs)

        super(AEModel, self).__init__(loss=loss, neurons=neurons,
                                      layers=None, activation=activation)

    def fast_fit(self, data, epochs=5, lr=1e-2, bs=16):
        data = self.tensor_from_np([data])

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        # torch_dataset = Data.TensorDataset(data, data)  # [:,1].unsqueeze(1))
        # loader = Data.DataLoader(
        #     dataset=torch_dataset, shuffle=True)
        fast_train(epochs=epochs, model=self, dataset=data)

    def get_projections(self, data):
        x = tensor_from_np([data])
        for layer in self.layers:
            x = self.activation(layer(x))
            if len(layer.weight) < data.shape[-1]:
                break
        return x.detach().numpy()


def test_base_model():
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    x = np.linspace(-10, 10, 3000)
    y = np.sin(x) + np.random.normal(0, 0.1, 3000)
    # x = np.dstack((x, np.sin(x)))[0]
    ds = np.dstack((x, y))[0]
    ds = StandardScaler().fit_transform(ds)

    x_val = np.linspace(-20, -15, 1000)
    # x_val = np.dstack((x_val, np.sin(x_val)))[0]
    y_val = np.sin(x_val) + np.random.normal(0, 0.1, 1000)
    ds_val = np.dstack((x_val, y_val))[0]
    ds_val = StandardScaler().fit_transform(ds_val)
    ds_val[:, 0] += 10

    model = NeuralNetwork(
        neurons=[1, 32, 16, 4, 1], loss=torch.nn.MSELoss(), activation=torch.tanh)

    model.fit(ds[:, 0], ds[:, 1], X_val=x_val, y_val=y_val,
              print_training=True, epochs=50, lr=1e-3)

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300, figsize=(5, 3))
    fig.tight_layout()

    ax1.set_title("Predictions on training set")
    ax2.set_title("Predictions on test set")

    ax1.plot(ds[:, 0], ds[:, 1], c='gray', alpha=0.3)
    preds_training = model.predict(X=ds[:, 0])
    ax1.plot(ds[:, 0], preds_training, label="Predictions", )

    ax2.plot(ds_val[:, 0], ds_val[:, 1], c='gray', alpha=0.3)
    preds_test = model.predict(X=ds_val[:, 0])
    ax2.plot(ds_val[:, 0], preds_test, label="Predictions", )

    ax1.legend()
    ax2.legend()
    plt.show()


# from files.NeuralRansac import *


def test2():

    ds = read_np_array(joinpath("datasets", "2d", "lines",
                       "with_outliers", "stair4.csv")).astype(float)
    gt = read_np_array(joinpath("datasets", "2d", "lines",
                       "with_outliers", "stair4_gt.csv"))
    gt = gt.astype(int).reshape(len(gt))

    # , activation=lambda x: x)
    model = AEModel(n_inputs=2, n_first_hidden=32, n_outputs=2)
    model.fit(ds, print_training=False, epochs=100)
    preds = model.predict(ds)
    plot(ds, c='gray', alpha=0.3)
    plot(preds)
    plt.show()


if __name__ == "__main__":
    ds, gt = load_dataset_by_name("circle5")
    nn = AEModel()
    nn.fit(ds, 100, print_training=True)
    projs = nn.get_projections(ds)
    print(projs.shape)
