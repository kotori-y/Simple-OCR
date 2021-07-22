import datetime

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from .datasets import VerifyCode
except ImportError:
    from datasets import VerifyCode

LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 32

ASCII = [chr(i) for i in range(48, 58)] + [chr(j) for j in range(97, 123)]
ASCII = dict(zip(range(10 + 26), ASCII))


class VerificationCodeCNN(nn.Module):
    def __init__(self, num_class=36, num_char=4):
        super(VerificationCodeCNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # [-1, 3, 35, 120]
            nn.Conv2d(3, 16, 3, padding=(1, 1)),  # [-1, 16, 35, 120]
            nn.MaxPool2d(2, 2),  # [-1, 16, 17, 230]
            nn.BatchNorm2d(16),  # [-1, 16, 17, 60]
            nn.ReLU(),

            nn.Conv2d(16, 64, 3, padding=(1, 1)),  # [-1, 64, 17, 60]
            nn.MaxPool2d(2, 2),  # [-1, 64, 8, 30]
            nn.BatchNorm2d(64),  # [-1, 64, 8, 30]
            nn.ReLU(),

            nn.Conv2d(64, 512, 3, padding=(1, 1)),  # [-1, 512, 8, 30]
            nn.MaxPool2d(2, 2),  # [-1, 512, 4, 15]
            nn.BatchNorm2d(512),  # [-1, 512, 4, 15]
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=(1, 1)),  # [-1, 512, 4, 15]
            nn.MaxPool2d(2, 2),  # [-1, 512, 2, 7]
            nn.BatchNorm2d(512),  # [-1, 512, 2, 7]
            nn.ReLU(),

        )
        self.fc = nn.Linear(512 * 2 * 7, self.num_class * self.num_char)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512 * 2 * 7)
        x = self.fc(x)
        return x


def train_step(model: nn.Module, features: torch.Tensor, y_true: torch.Tensor):
    # training mode
    model.train()

    # clear gradient
    model.opt.zero_grad()

    # forward
    y_pred = model(features)
    loss = model.loss_func(y_pred, y_true)
    metric, _, _ = model.metric_func(y_pred, y_true)

    # BP
    loss.backward()
    model.opt.step()

    return loss.item(), metric


def valid_step(model: nn.Module, features: torch.Tensor, y_true: torch.Tensor):
    # validation mode
    model.eval()

    # close
    with torch.no_grad():
        y_pred = model(features)
        loss = model.loss_func(y_pred, y_true)
        metric, _, _ = model.metric_func(y_pred, y_true)

    return loss.item(), metric


def calculate_acc(output, target, mode=None):
    output, target = output.view(-1, 36), target.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    p, t = [], []
    for i, j in zip(output, target):
        res = 1 if torch.equal(i, j) else 0
        correct_list.append(res)
        if mode == "predicting":
            p.append("".join([ASCII[x] for x in i.numpy()]))
            t.append("".join([ASCII[x] for x in j.numpy()]))
    acc = sum(correct_list) / len(correct_list)
    return acc, p, t


def train_model(model: nn.Module, epochs, dl_train, dl_valid, log_step_freq):
    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])

    print("Start ... ")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % nowtime)

    for epoch in range(1, epochs + 1):

        loss_sum = 0
        metric_sum = 0
        step = 1

        val_loss_sum = 0
        val_metric_sum = 0
        val_step = 1

        for step, (features, y_true) in enumerate(dl_train, 1):

            loss, metric = train_step(model, features, y_true)
            loss_sum += loss
            metric_sum += metric
            if step % log_step_freq == 0:
                print(
                    ("[step = %d] loss: %.6f, " + metric_name + ": %.6f") %
                    (step, loss_sum / step, metric_sum / step))

        for val_step, (features, y_true) in enumerate(dl_valid, 1):
            val_loss, val_metric = valid_step(model, features, y_true)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        info = (epoch, loss_sum / step, metric_sum / step,
                val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info

        # 打印epoch级别日志
        print(
            ("\nEPOCH = %d, loss = %.6f, " + metric_name +
             "  = %.6f, val_loss = %.6f, " + "val_" +
             metric_name + "= " "%.6f")
            % info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)

    print('Finished Training...')

    return dfhistory


def build_dataloader(path, batch_size, shuffle=True):
    ds = VerifyCode(path, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl


if __name__ == "__main__":
    codeOCR = VerificationCodeCNN()
    codeOCR.loss_func = nn.MultiLabelSoftMarginLoss()
    codeOCR.metric_func = lambda y_pred, y_true: calculate_acc(y_pred, y_true)
    codeOCR.metric_name = "ACC"
    codeOCR.opt = torch.optim.Adam(codeOCR.parameters(), lr=LR)

    dl_train = build_dataloader("../data/training", batch_size=BATCH_SIZE)
    dl_valid = build_dataloader("../data/validation", batch_size=BATCH_SIZE)

    dfhistory = train_model(codeOCR, EPOCHS, dl_train, dl_valid, 100)

    print("DONE")
