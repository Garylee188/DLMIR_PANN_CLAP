import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

split = 'val'
title_split = 'Testing'
df = pd.read_csv(f'G:/dlmir_result_1219/result_{split}.csv')
all_classes = sorted(set(df['gt']))
print(all_classes)

data_dir = r'G:\dlmir_dataset\emo\emo\emotion'
emo = sorted(os.listdir(data_dir))
emo_sub = [emo[i] for i in all_classes]

specific_class_pred = {}  # to see zero-shot class prediction
each_class_correct, each_class_total, each_class_acc = {}, {}, {}
for c in emo_sub:
    each_class_correct[c] = 0
    each_class_total[c] = 0
    specific_class_pred[c] = 0

for i in range(len(df['gt'])):
    if emo[df['gt'][i]] == "Gentle_tt":
        specific_class_pred[emo[df['pred'][i]]] += 1

    if df['gt'][i] == df['pred'][i]:
        each_class_correct[emo[df['gt'][i]]] += 1
    each_class_total[emo[df['gt'][i]]] += 1

print(each_class_correct)
print()
print(each_class_total)
print()
for key, val in each_class_correct.items():
    each_class_acc[key] = 100.0 * val / each_class_total[key]
print(each_class_acc)
print()
print(specific_class_pred)
print()

GROUP_I = ['Fierce_tt','Visceral_tt']
GROUP_P = ['Celebratory_tt', 'Hopeful_tt', 'Reverent_tt']
GROUP_C = ['Delicate_tt', 'Indulgent_tt', 'Nostalgic_tt', 'Rebellious_tt', 'Refined_tt', 'Rollicking_tt']
GROUP_U = ['Greasy_tt', 'Lush_tt', 'Soft_tt']
GROUP_N = ['Bitter_tt', 'Gloomy_tt', 'Menacing_tt', 'Outrageous_tt', 'Sarcastic_tt', 'Spooky_tt']
group_i, group_p, group_c, group_u, group_n = [], [], [], [], []

for emotion, acc in each_class_acc.items():
    if emotion in GROUP_I:
        group_i.append(acc)
    elif emotion in GROUP_P:
        group_p.append(acc)
    elif emotion in GROUP_C:
        group_c.append(acc)
    elif emotion in GROUP_U:
        group_u.append(acc)
    elif emotion in GROUP_N:
        group_n.append(acc)

Group_Result = {
    'I': sum(group_i) / len(group_i),
    'P': sum(group_p) / len(group_p),
    'C': sum(group_c) / len(group_c),
    'U': sum(group_u) / len(group_u),
    'N': sum(group_n) / len(group_n)
}
print(Group_Result)
mean_acc = 0
for key, val in Group_Result.items():
    mean_acc += val
print(mean_acc / 4)
# x = np.arange(1, 22) * 2
# plt.barh(x, list(specific_class_pred.values()), tick_label=emo_sub, color='gray', height=1)
# # plt.title(f'Accuracy of {title_split} set')
# plt.title(f'Zero Shot Result')
# plt.xlabel('# of songs')
# plt.tight_layout()
# plt.savefig(f'G:/dlmir_result_1219/{split}_zero_shot.png')
# plt.show()
# plt.close()

# x = np.arange(1, 6) * 2
# plt.barh(x, list(Group_Result.values()), tick_label=['I','P','C','U','N'], color='gray', height=1)
# plt.title(f'Accuracy of {title_split} Group set')
# plt.xlabel('Accuracy (%)')
# plt.tight_layout()
# plt.savefig(f'G:/dlmir_result_1219/{split}_group_acc.png')
# plt.show()
# plt.close()