import os
import pandas as pd
import random

random.seed(0)

sub_labels = random.sample(range(190), k=20)

SubDataset = {}
SubDataset['filename'] = []
SubDataset['label'] = []
df = pd.read_csv(r'G:\dlmir_dataset\test_2.csv')
for ids, label in enumerate(df['label']):
    if label in sub_labels:
        SubDataset['filename'].append(df['filename'][ids])
        SubDataset['label'].append(label)
pd.DataFrame(SubDataset).to_csv('./20_test.csv', index=False)
