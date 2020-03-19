import torch
import numpy as np
from datasets.datasets import Datasets
from models.models import DeepModel

"""
datasets = Datasets("Bird Song")
cardinality = datasets.get_cardinality_possible_partial_set()

datasets.set_mode('train')
print(datasets.get_dims)
"""

"""
X, y_partial, y, idx = datasets[0]

print(cardinality[idx], cardinality[idx] == np.count_nonzero(y_partial))
print('y_partial', y_partial)

example_label_weights = []
for card in cardinality:
    example_label_weights.append(torch.softmax(torch.autograd.Variable(torch.ones(card)), 0))

print(example_label_weights[idx])
print(y_partial.shape)
print(y_partial.nonzero())

print(np.array(y_partial.nonzero()))

print('cardinality[i]', cardinality[idx])


label_idx = np.random.choice(np.arange(cardinality[idx]))
label = np.array(y_partial.nonzero())[0][label_idx] # Lower Loop Target
label_weight = example_label_weights[idx][label_idx] # Upper Loop Parameter

print(label_idx, label, label_weight)

candidate_idx = torch.Tensor(y_partial).nonzero(as_tuple=True)
non_candidate_idx = torch.Tensor(1 - y_partial).nonzero(as_tuple=True)

scores = torch.softmax(torch.rand(13), 0)
print(torch.sum(scores[candidate_idx]))
print(torch.sum(scores[non_candidate_idx]))
"""

model = DeepModel(16, 4).cuda()
out = model(torch.ones(16, 32).cuda())
print(out.size())
