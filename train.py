import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import gc
import pickle
import time

debug = False

def decide(output):
    return np.argmax(output.detach().cpu().numpy(), axis=1)

def validate(model, val_loader, objective, batch_size, img_size=300, cuda=False):
    val_loss_list = []
    val_acc_list = []
    if cuda:
        model = model.cuda()
    with torch.no_grad():
        for val_x, val_y_truth in val_loader:
            if cuda:
                val_x, val_y_truth = val_x.cuda(), val_y_truth.cuda()
            gc.collect()
            val_y_hat = model(val_x)
            val_loss_list.append(objective(val_y_hat, val_y_truth.squeeze().long()).item())
            val_pred = decide(val_y_hat)
            val_acc_list.append(np.sum(val_pred == val_y_truth.detach().cpu().numpy()) / (img_size * img_size * batch_size))
            print("batch")
    return val_loss_list, val_acc_list

def train(model, train_loader, val_loader, objective, optimizer, epochs, batch_size, img_size=300, cuda=False):
    start_time = time.time()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    if cuda:
        model = model.cuda()

    for epoch in range(epochs):
        # validation
        if not debug:
            val_loss, val_acc = validate(model, val_loader, objective, batch_size, img_size=img_size, cuda=cuda)
            val_losses.append(sum(val_loss) / float(len(val_loss)))
            val_accs.append(sum(val_acc) / float(len(val_acc)))
            print('epoch {} validation: val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch, val_losses[-1], val_accs[-1]))

        # training
        batch = 0
        for x, y_truth in train_loader:
            if cuda:
                x, y_truth = x.cuda(), y_truth.cuda()
            gc.collect()
            # learn
            optimizer.zero_grad()
            y_hat = model(x)
            loss = objective(y_hat, y_truth.squeeze().long())
            pred = decide(y_hat)
            percent = np.sum(pred == y_truth.squeeze().detach().cpu().numpy()) / (img_size * img_size * batch_size)
            train_losses.append(loss.item())
            train_accs.append(percent)
            loss.backward()
            optimizer.step()
            batch += 1

        print('epoch {} training: last train loss: {:.4f}, last train acc: {:.4f}'.format(epoch, train_losses[-1],
                                                                                      train_accs[-1]))    # final validation
    if not debug:
        val_loss, val_acc = validate(model, val_loader, objective, batch_size, img_size=img_size, cuda=cuda)
        val_losses.append(sum(val_loss) / float(len(val_loss)))
        val_accs.append(sum(val_acc) / float(len(val_acc)))
        print('final validation: val_loss: {:.4f}, val_acc:{:.4f}'.format(val_losses[-1], val_accs[-1]))

    print("--- Training took %s hours ---" % ((time.time() - start_time) / 3600))
    return train_losses, train_accs, val_losses, val_accs

def save_train_results(model, results, dir):
    """
    :param model:
    :param results:
        - model_name
        - epochs
        - learn_rate
        - batch_size
        - train_losses, train_accs, val_losses, val_accs
    :param dir:
    :return:
    """
    torch.save(model, dir + results['model_name'] + ".pt")
    f = open(dir + results['model_name'] + "_stats.pkl", "wb")
    pickle.dump(results, f)


