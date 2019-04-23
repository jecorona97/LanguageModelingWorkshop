import os
import json
from tqdm import tqdm


def read_fb_messages(dump_path="data/messages/inbox/"):
    messages = []
    for file in tqdm(os.listdir(dump_path)):
        if os.path.isdir(dump_path + file):
            with open(dump_path + file + "/message_1.json", 'r') as f:
                data = json.load(f)
            for msg in data["messages"]:
                if "content" in msg:
                    messages.append((msg["content"]))
    return messages 