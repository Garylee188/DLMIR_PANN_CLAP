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

# Set path
data_dir = r'G:\dlmir_dataset\emo\emo\emotion'
train_csv = './20_train.csv'
val_csv = './20_val.csv'
json_dir = r'G:\dlmir_dataset\emo\emo\token_json_2'

# Set hyper parameters
lr = 4e-5
epochs = 20
batch_size = 128

# Prepare data
train_data = EmoDataset(data_dir, train_csv, json_dir)
val_data = EmoDataset(data_dir, val_csv, json_dir)
train_loader = DataLoader(train_data, batch_size, shuffle=True)
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
model_repo = "microsoft/msclap"
model_name = {
    '2022': 'CLAP_weights_2022.pth',
}
model_fp = hf_hub_download(model_repo, model_name['2022'])
model_state_dict = torch.load(model_fp, map_location=device)['model']
clap_model.load_state_dict(model_state_dict, strict=False)

# Change original classes number to 20 classes, also initialize fc layer weight and bias.
clap_model.audio_encoder.base.fc_audioset = nn.Linear(2048, 20, bias=True)
nn.init.xavier_uniform_(clap_model.audio_encoder.base.fc_audioset.weight)
nn.init.zeros_(clap_model.audio_encoder.base.fc_audioset.bias)
clap_model.to(device)

# print(clap_model)

save_model_name = 'clap_cnn14_bert_new_caption.pth'
save_model_path = r'G:\dlmir_result_1223'
os.makedirs(save_model_path, exist_ok=True)

minLoss = 1000.
optimizer = optim.Adam(clap_model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

def log_cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)


criterion = nn.CrossEntropyLoss()
# Train
for epoch in range(epochs):
    train_loss_batch = 0.0
    clap_model.train()

    with tqdm(total=len(train_loader)) as train_bar:
        for batch in train_loader:
            waveform, token = batch['waveform'].to(device), batch['token'].to(device)
            token['input_ids'] = token['input_ids'].squeeze()
            token['token_type_ids'] = token['token_type_ids'].squeeze()
            token['attention_mask'] = token['attention_mask'].squeeze()

            text_embeddings, audio_embeddings, logit_scale = clap_model(waveform, token)
            audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
            text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
            
            # cosine similarity as logits, shape = [batch_size, batch_size]
            logits_per_audio = logit_scale * audio_embeddings @ text_embeddings.t()
            logits_per_text = logits_per_audio.t()
            
            batch_size = audio_embeddings.shape[0]
            # max value on diag.
            labels = torch.arange(batch_size, device=device).long()
            loss = (
                criterion(logits_per_audio, labels) +
                criterion(logits_per_text, labels)
            ) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix({'Train Loss': loss.item()}, refresh=True)
            train_bar.update(1)
            train_loss_batch += loss.detach().cpu().item()
        train_loss_epoch = train_loss_batch / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_epoch}')


    val_loss_batch = 0.0
    clap_model.eval()
    with tqdm(total=len(val_loader)) as val_bar:
        with torch.no_grad():
            for batch in val_loader:
                waveform, token = batch['waveform'].to(device), batch['token'].to(device)
                token['input_ids'] = token['input_ids'].squeeze()
                token['token_type_ids'] = token['token_type_ids'].squeeze()
                token['attention_mask'] = token['attention_mask'].squeeze()
                
                text_embeddings, audio_embeddings, logit_scale = clap_model(waveform, token)
                audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
                text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
                
                # cosine similarity as logits, shape = [batch_size, batch_size]
                logits_per_audio = logit_scale * audio_embeddings @ text_embeddings.t()
                logits_per_text = logits_per_audio.t()
                
                batch_size = audio_embeddings.shape[0]
                # max value on diag.
                labels = torch.arange(batch_size, device=device).long()
                loss = (
                    criterion(logits_per_audio, labels) +
                    criterion(logits_per_text, labels)
                ) / 2
                # loss = (
                #     criterion(logits_per_audio, labels) +
                #     criterion(logits_per_text, labels)
                # )

                # if (epoch + 1) % 5 == 0:  # save model every 5 epoch
                #     torch.save(clap_model.state_dict(), os.path.join(save_model_path, f'{epoch+1}_{save_model_name}'))

                if loss.item() < minLoss:
                    minLoss = loss.item()
                    torch.save(clap_model.state_dict(), os.path.join(save_model_path, f'best_{save_model_name}'))
                
                val_bar.set_postfix({'Val Loss': loss.item()}, refresh=True)
                val_bar.update(1)
                val_loss_batch += loss.detach().cpu().item()

    ### New add for accuracy estimation
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as val_bar:
            correct = 0
            total = 0
            for batch in val_loader:
                waveform = batch['waveform'].to(device)#,batch['token']
                token,keylist = val_data.get_uniform_token()
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
                # batch_size = audio_embeddings.shape[0]
                # max value on diag.
                # labels = torch.arange(batch_size, device=device).long()
                # print(keylist[pred.cpu().numpy()])
                # print(batch['target'].cpu().numpy())
                correct += (keylist[pred.cpu().numpy()] == batch['target'].cpu().numpy()).sum().item()
                total += pred.shape[0]
            print(f"Validation Accuracy: {100 * correct / total}%")

        
        val_loss_epoch = val_loss_batch / len(val_loader)
        scheduler.step(val_loss_epoch)

    print(f'Epoch {epoch+1}/{epochs}, Val Loss: {val_loss_epoch}')