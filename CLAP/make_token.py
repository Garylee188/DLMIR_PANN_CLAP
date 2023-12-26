import json
import os

root_dir = r'G:\dlmir_dataset\emo\emo\emotion'
json_dir = r'G:\dlmir_dataset\emo\emo\token_json_3'

emotions = sorted(os.listdir(root_dir))

for emo in emotions:
    emo_tokens_dict = {}
    tag = emo.split("_")[0].lower()
    text = f"if not a {tag} song, no score"
    emo_tokens_dict["text"] = [text]
    emo_tokens_dict["tag"] = [tag]

    for song in os.listdir(os.path.join(root_dir, emo)):
        # if song.endswith(".json"):
        #     os.remove(os.path.join(root_dir, emo, song))
        token_json = song[:-4] + ".json"
        os.makedirs(os.path.join(json_dir, emo), exist_ok=True)
        json_file = os.path.join(json_dir, emo, token_json)

        json_str = json.dumps(emo_tokens_dict)
        with open(json_file, "w") as file:
            file.write(json_str)