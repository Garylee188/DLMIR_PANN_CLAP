from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
import librosa
import json
from transformers import BertTokenizer

np.random.seed(0)

class EmoDataset(Dataset):
    def __init__(self, song_dir, song_csv, json_dir):
        self.song_dir = song_dir
        self.song_file = pd.read_csv(song_csv)['filename']
        self.song_label = pd.read_csv(song_csv)['label']
        self.json_dir = json_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_uniform_token(self):
        unique_list = []
        final_list = []
        key_list= []
        for k in self.song_label[:self.__len__()]:
            # print(k)
            each =  sorted(os.listdir(self.song_dir))[k]
            # check if exists in unique_list or not
            if each not in unique_list:
                unique_list.append(each)
                key_list.append(k)
                # print(each)
                final_list.append(f"if not a {each} song, no score")
        # print(len(unique_list))
        return self.tokenizer(final_list,
                            return_tensors='pt', 
                            padding='max_length',
                            truncation=True, max_length=50),np.array(key_list)



    def __getitem__(self, index):
        emotion = sorted(os.listdir(self.song_dir))[self.song_label[index]]
        y, sr = librosa.load(os.path.join(self.song_dir, emotion, self.song_file[index]),sr=44100)
        start_time = np.random.uniform(0, len(y) - 10 * sr)
        song_clip = y[int(start_time):int(start_time + 10 * sr)]
        json_file = os.path.join(self.json_dir, emotion, self.song_file[index][:-4] + '.json')
        with open(json_file, 'r') as file:
            json_data = json.load(file)
        text = json_data['text']

        if len(song_clip) != 441000:
            emotion = sorted(os.listdir(self.song_dir))[self.song_label[0]]
            y, sr = librosa.load(os.path.join(self.song_dir, emotion, self.song_file[0]),sr=44100)
            start_time = np.random.uniform(0, len(y) - 10 * sr)
            song_clip = y[int(start_time):int(start_time + 10 * sr)]
            json_file = os.path.join(self.json_dir, emotion, self.song_file[0][:-4] + '.json')
            with open(json_file, 'r') as file:
                json_data = json.load(file)
            text = json_data['text']

        
        token = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
        output = {'waveform': song_clip,
                  'text': text,
                  'token': token,
                  'target': self.song_label[index]}
        return output
    
    def __len__(self):
        return len(self.song_file)
    
if __name__=="__main__":
    data_dir = r'G:\dlmir_dataset\emo\emo\emotion'
    csv_file = r'D:\CLAP\20_train.csv'
    json_dir = r'G:\dlmir_dataset\emo\emo\token_json'

    # for emo in os.listdir(data_dir):
    #     print(emo.split('_')[0])
    train_dataset = EmoDataset(data_dir, csv_file, json_dir)
    # a,b = train_dataset.ge
    # print(b)
    for i in range(2):
        # print(train_dataset.__getitem__(i)['token']['input_ids'].size(1))
        # if train_dataset.__getitem__(i)['token']['input_ids'].size(1) != 7:
        #     print(train_dataset.__getitem__(i)['text'])
        print(train_dataset.__getitem__(i)['waveform'])