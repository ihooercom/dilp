import json
import sys
import random

file_path = sys.argv[1]
dst_dir = sys.argv[2]
with open(file_path, 'r') as f:
    data = json.load(f)
instances = data
random.shuffle(instances)

train_num = int(0.95 * len(instances))
val_num = int(0.05 * len(instances))
train_data = instances[:train_num]
val_data = instances[-val_num:]
print('train_num: ', len(train_data))
print('val_num: ', len(val_data))
with open(dst_dir + '/train.json', 'w') as f:
    json.dump(train_data, f, indent=4)
with open(dst_dir + '/val.json', 'w') as f:
    json.dump(val_data, f, indent=4)