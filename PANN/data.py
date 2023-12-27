import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
from tqdm.auto import tqdm

def move_files(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files = os.listdir(source_folder)
    for file in files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)
        # print(f"Moved {file} to {destination_folder}")

emo_folder_path = '/home/patrickwei/audioset_tagging_cnn/DLMIR_Final/emotion'

num = []
trainData = {}
valData = {}
testData = {}
trainData['filename'] = []
trainData['label'] = []
valData['filename'] = []
valData['label'] = []
testData['filename'] = []
testData['label'] = []
cant_load_dir = r'G:\dlmir_dataset\emo\emo\cant_load'

for idx, emo in enumerate(tqdm(sorted(os.listdir(emo_folder_path)))):
    # print(idx, emo)
    songs = os.listdir(os.path.join(emo_folder_path, emo))
    cant_load_songs = []
    for song in songs:
        file_path = os.path.join(emo_folder_path, emo, song)
        try:
            y, sr = librosa.load(file_path)
        except:
            # print(file_path)
            os.makedirs(os.path.join(cant_load_dir, emo), exist_ok=True)
            # shutil.move(file_path, os.path.join(cant_load_dir, emo))  # , os.path.join(cant_load_dir, emo, song)
            cant_load_songs.append(file_path)
    for p in cant_load_songs:
        shutil.move(p, os.path.join(cant_load_dir, emo))
    songs = os.listdir(os.path.join(emo_folder_path, emo))
    labels = [idx] * len(songs)
    
    train_data, val_test_data, train_label, val_test_label = train_test_split(songs, labels, test_size=0.2, random_state=0)
    val_data, test_data, val_label, test_label = train_test_split(val_test_data, val_test_label, test_size=0.5, random_state=0)

    trainData['filename'].extend(train_data)
    trainData['label'].extend(train_label)
    valData['filename'].extend(val_data)
    valData['label'].extend(val_label)
    testData['filename'].extend(test_data)
    testData['label'].extend(test_label)


pd.DataFrame(trainData).to_csv('./train_3.csv', index=False)
pd.DataFrame(valData).to_csv('./val_3.csv', index=False)
pd.DataFrame(testData).to_csv('./test_3.csv', index=False)
