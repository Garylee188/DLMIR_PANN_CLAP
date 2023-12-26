import os
import json

emo_path = r'G:\dlmir_dataset\emo\emo\emotion'
class_label_dict = {}
for label, cls in enumerate(sorted(os.listdir(emo_path))):
    # print(label, cls)
    class_label_dict[cls.split('_')[0].lower()] = label

json_str = json.dumps(class_label_dict)
with open("class_label.json", "w") as json_file:
    json_file.write(json_str)