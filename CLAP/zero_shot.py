import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
from msclap.models import clap
from dataset import EmoDataset
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm
from huggingface_hub.file_download import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict

# Set path
data_dir = r'G:\dlmir_dataset\emo\emo\emotion'
train_csv = './20_train.csv'
val_csv = './20_val.csv'
json_dir = r'G:\dlmir_dataset\emo\emo\token_json_2'
save_path = 'G:/dlmir_result_1219'
save_model_path = r'G:\dlmir_result_1219\best_clap_cnn14_bert_new_caption.pth'
# Set hyper parameters
lr = 1e-3
epochs = 40
batch_size = 64

# Prepare data
# train_data = EmoDataset(data_dir, train_csv, json_dir)
val_data = EmoDataset(data_dir, val_csv, json_dir)
# train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=False)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Build model
clap_model = clap.CLAP(
    # audio
    audioenc_name='Cnn14',
    sample_rate=44100, 
    window_size=1024, 
    hop_size=320, 
    mel_bins=64, 
    fmin=50, 
    fmax=14000, 
    classes_num=527, 
    out_emb=2048,
    # text
    text_model='bert-base-uncased',
    transformer_embed_dim=768,
    # common
    d_proj=1024,
)
# model_repo = "microsoft/msclap"
# model_name = {
#     '2022': 'CLAP_weights_2022.pth',
# }
# model_fp = hf_hub_download(model_repo, model_name['2022'])
# model_state_dict = torch.load(model_fp, map_location=device)['model']


# Change original classes number to 20 classes, also initialize fc layer weight and bias.
clap_model.audio_encoder.base.fc_audioset = nn.Linear(2048, 20, bias=True)
nn.init.xavier_uniform_(clap_model.audio_encoder.base.fc_audioset.weight)
nn.init.zeros_(clap_model.audio_encoder.base.fc_audioset.bias)
model_state_dict = torch.load(save_model_path)
clap_model.load_state_dict(model_state_dict, strict=False)

clap_model.to(device)
clap_model.eval()
# print(clap_model)


def log_cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)


# eval
with torch.no_grad():
    correct = 0
    total = 0
    all_pred, all_gt = [], []
    for batch in val_loader:
        waveform = batch['waveform'].to(device)#,batch['token']
        token, keylist = val_data.get_uniform_token()
        token = token.to(device)
        token['input_ids'] = token['input_ids'].squeeze()
        token['token_type_ids'] = token['token_type_ids'].squeeze()
        token['attention_mask'] = token['attention_mask'].squeeze()
        # print(token.shape)
        
        text_embeddings, audio_embeddings, logit_scale = clap_model(waveform, token)
        audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
        text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
        # cosine similarity as logits, shape = [batch_size, batch_size]
        similarity = (100.0 * audio_embeddings @ text_embeddings.T)#.softmax(dim=-1)
        # print(similarity.shape)
        # print('____________________________')
        pred = torch.argmax(similarity,dim=1)
        all_pred.extend(keylist[pred.cpu().numpy()])
        all_gt.extend(batch['target'].tolist())
        # batch_size = audio_embeddings.shape[0]
        # max value on diag.
        # labels = torch.arange(batch_size, device=device).long()
        # print(keylist[pred.cpu().numpy()])
        # print(batch['target'].cpu().numpy())
        correct += (keylist[pred.cpu().numpy()] == batch['target'].cpu().numpy()).sum().item()
        total += pred.shape[0]
    classes_dist = {
        'pred': all_pred,
        'gt': all_gt
    }
    pd.DataFrame(classes_dist).to_csv(f'{save_path}/result_val.csv')
    print(f"Validation Accuracy: {100 * correct / total}%")


    pred_counter = Counter(all_pred)
    gt_counter = Counter(all_gt)
    if len(pred_counter.keys()) != len(gt_counter.keys()):
        miss_class = set(gt_counter.keys()) - set(pred_counter.keys())
        for s in miss_class:
            pred_counter[int(s)] = 0
    od_pred = {key: pred_counter[key] for key in sorted(pred_counter)}
    od_gt = {key: gt_counter[key] for key in sorted(gt_counter)}
    
    emo = sorted(os.listdir(data_dir))
    emo_20 = [emo[i] for i in od_gt.keys()]
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # # gt
    # axes[0].barh(list(gt_counter.keys()), list(gt_counter.values()))
    # axes[0].set_title('GT Pie Chart')

    # # pred
    # axes[1].barh(list(pred_counter.keys()), list(pred_counter.values()))
    # axes[1].set_title('Pred Pie Chart')

    x = np.arange(1, 21) * 2
    x2 = [a-0.2 for a in x]
    # plt.barh(x, list(gt_counter.values()), tick_label=emo_20, color='b', height=0.4, align='edge')
    plt.barh(x, list(pred_counter.values()), tick_label=emo_20, color='gray', height=1)
    plt.title('Class Distribution of Prediction')

    plt.tight_layout()
    plt.savefig(f'{save_path}/result_val.png')
    plt.show()