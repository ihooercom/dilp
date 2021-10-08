import torch
import torch.nn.functional as F
import numpy as np

model_path = 'exps/block_world/pt/model_state_e9_b0.th'
model = torch.load(model_path)

for n in model:
    if n.startswith('rule_weights'):
        p = n[len('rule_weights_'):-2]
        print(p)
        w = model[n]
        scores = F.softmax(w)
        print(scores[scores > 0.01])
#         w = torch.where(scores > 1e-3, 0.0, -4.0)
#         scores = F.softmax(w)
#         print(scores[scores > 0.01])
#         model[n] = w
# torch.save(model, model_path)