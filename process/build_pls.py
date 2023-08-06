import json

import numpy as np
import torch

from unixcoder import UniXcoder

model = UniXcoder("unx")
device = torch.device("cuda")
model.to(device)

for x in ["train", "test", "valid"]:

    array_list = []

    project_name = "ffmpeg/"
    task = x

    with open(f"../data/raw/{project_name + task}.jsonl", 'r') as f:
        func_list = f.readlines()

    for i in range(0, len(func_list)):
        func = json.loads(func_list[i].replace("\n", ""))["func"].replace("\n", "")
        tokens_ids = model.tokenize([func], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings, max_func_embedding = model(source_ids)
        max_func_embedding = max_func_embedding.cpu().detach().numpy()
        array_list.append(max_func_embedding)
        print(len(array_list))

    np.save(f"../data/dataset/{task}_emb.npy", np.stack(array_list))
