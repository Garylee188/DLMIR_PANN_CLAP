from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import librosa
from pydub import AudioSegment

np.random.seed(0)

class EmoDataset(Dataset):
    def __init__(self, song_dir, song_csv):
        self.song_dir = song_dir
        self.song_file = pd.read_csv(song_csv)['filename']
        self.song_label = pd.read_csv(song_csv)['label']
        self.desired_sr = 22050
    
    def get_label(self):
        all = sorted(os.listdir(self.song_dir))
        label = []
        for each in self.song_label[:self.__len__()]:
            if all[each] not in label:
                label.append(all[each])
                
        return label
        
        
    def __getitem__(self, index):
        print(sorted(os.listdir(self.song_dir)))
        try:
            emotion = sorted(os.listdir(self.song_dir))[self.song_label[index]]
            # y, sr = librosa.load(os.path.join(self.song_dir, emotion, self.song_file[index]),sr=44100)
            # print(os.path.join(self.song_dir, emotion, self.song_file[index]))
            audio = AudioSegment.from_mp3(os.path.join(self.song_dir, emotion, self.song_file[index]))
            audio = audio.set_frame_rate(self.desired_sr)
            y = np.array(audio.get_array_of_samples()).astype(float)
        except:
            emotion = sorted(os.listdir(self.song_dir))[self.song_label[2]]
            # y, sr = librosa.load(os.path.join(self.song_dir, emotion, self.song_file[index]),sr=44100)
            audio = AudioSegment.from_mp3(os.path.join(self.song_dir, emotion, self.song_file[2]))
            audio = audio.set_frame_rate(self.desired_sr)
            y = np.array(audio.get_array_of_samples()).astype(float)
        start_time = np.random.uniform(0, len(y) - 10 * self.desired_sr)
        song_clip = y[int(start_time):int(start_time + 10 * self.desired_sr)]
        if len(song_clip) != 441000//2:
            emotion = sorted(os.listdir(self.song_dir))[self.song_label[2]]
            # y, sr = librosa.load(os.path.join(self.song_dir, emotion, self.song_file[index]),sr=44100)
            audio = AudioSegment.from_mp3(os.path.join(self.song_dir, emotion, self.song_file[2]))
            audio = audio.set_frame_rate(self.desired_sr)
            y = np.array(audio.get_array_of_samples()).astype(np.float32)
            start_time = np.random.uniform(0, len(y) - 10 * self.desired_sr)
            song_clip = y[int(start_time):int(start_time + 10 * self.desired_sr)]
        output = {'waveform':song_clip,
                #   'filename':self.song_file[index],
                  'target':self.song_label[index]}
        return output
    
    def __len__(self):
        return len(self.song_file)
    
if __name__=="__main__":
    train_dataset = EmoDataset('emotion',"20_val.csv")
    print(train_dataset.get_label())
    # for i in range(100):
    #     print(train_dataset.__getitem__(0)['waveform'].shape)
