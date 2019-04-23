import os
import json
from tqdm import tqdm
import numpy as np
from keras.preprocessing.sequence import pad_sequences

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


def load_reuters_dataset(path="data/", dset=["train", "dev", "test"]):
    class Data: pass
    data = Data()

    for ds in tqdm(dset):
        with open(path + "reuters." + ds + ".txt", 'r', errors="replace") as f:
            data.__dict__[ds] = []
            for line in f:
                data.__dict__[ds].append([w for w in line.replace('\n', '').split(' ') if len(line) > 3] + ["<EOS>"])
    
    return data


def create_context_target_pairs(corpus):
    context = []
    target = []
    for sentence in corpus:
        for i in range(1, len(sentence)):
            subsequence = sentence[:i].copy()
            context.append(subsequence)
            target.append(sentence[i])
    return context, target

def generate_sentence(model, tokenizer, maxlen):
    sentence = []
    while (len(sentence) < maxlen):
        tokens = tokenizer.texts_to_sequences([sentence])
        input_toks = np.array(pad_sequences(tokens, maxlen=maxlen-1, padding='post'))
        yhat = model.predict(input_toks)[0]
        word_i = np.random.choice(yhat.shape[0], p=yhat)
        word = tokenizer.index_word[word_i]
        sentence.append(word)
        if word == "<EOS>":
            break
    return sentence