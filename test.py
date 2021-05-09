from absl import app
from absl import flags
import os
import torch
import torch.utils.data.dataloader
import tqdm
from datetime import datetime


def test_model(model, test_data):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=3)
    correct = 0
    total = 0
    for point in test_loader:
        sample, target = point
        output = model.forward(sample)
        prediction = torch.max(output, dim=1).indices
        correct += (prediction == target).float().sum()
        total += target.shape[0]
    accuracy = correct/total * 100
    timestamp = datetime.now()
    with open('accuracy' + str(model.name) + '.txt', 'a') as file:
        file.write('Date: {} Accuracy: {}'.format(timestamp, accuracy))
    return accuracy
