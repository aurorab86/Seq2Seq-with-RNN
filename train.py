import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from dataset import *
from model import *
from IPython.display import clear_output

import matplotlib.pyplot as plt


def plot_loss(losses):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss & Epochs')
    plt.legend()
    plt.grid()
    plt.show()

def model_train(epochs, batch_size, optimizer, model, losses, loss_average_list):
    for epoch in range(epochs):
        _, x, label = generate_data(batch_size=batch_size)

        x = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        optimizer.zero_grad()

        output = model(x)

        loss = nn.CrossEntropyLoss()(output.view(-1, 12), label.view(-1))
        losses.append(loss.item())

        loss.backward()

        optimizer.step()

        if epoch % batch_size == 0 and epoch > 0:
            loss_average = np.mean(losses)
            loss_average_list.append(loss_average)
            plot_loss(loss_average_list)
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
    print(f"Final loss average: {loss_average_list[-1]}")

