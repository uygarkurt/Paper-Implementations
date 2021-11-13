import torch.nn as nn
import torch.optim as optim
import torch

def train(sentences, input_batch, target_batch, model, optimizer, criterion):
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    return model
