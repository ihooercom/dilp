import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from itertools import product
import json

model_path = 'exps/block_world/pt/model_state_e13_b0.th'
model = torch.load(model_path)
all_clauses = torch.load('exps/block_world/rules_4.th')['all_clauses']
all_clauses = {p.name: all_clauses[p] for p in all_clauses}

cand_rules = defaultdict(list)
clauses_lst = []
for n in model:
    if n.startswith('rule_weights'):
        p = n[len('rule_weights_'):-2]
        idx = int(n.split('.')[-1])
        w = model[n]
        scores = F.softmax(w)
        I = (scores > 0.006).nonzero().view(-1)
        clauses = all_clauses[p][idx]
        cand_clauses = [str(clauses[i]) for i in I]
        cand_rules[p].append(cand_clauses)
        clauses_lst.append(cand_clauses)
cand_rules_list = list(product(*clauses_lst))
print('candidate rules num: ', len(cand_rules_list))

with open('exps/block_world/candidate_rules_from_dilp.json', 'w') as f:
    json.dump([r[4:] + r[:4] for r in cand_rules_list], f, indent=4)