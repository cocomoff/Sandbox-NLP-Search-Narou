import sentencepiece
from transformers import AlbertTokenizer, AlbertForPreTraining
import numpy as np
import torch


if __name__ == '__main__':
    query = '悪徳令嬢'
    tokenizer = AlbertTokenizer.from_pretrained('ALINEAR/albert-japanese-v2')
    model = AlbertForPreTraining.from_pretrained('ALINEAR/albert-japanese-v2')

    input_ids= torch.tensor(tokenizer.encode(query, add_special_tokens=True)).unsqueeze(0) 
    embed_q = torch.mean(model(input_ids, output_hidden_states=True).hidden_states[12], 1)
    
    # 各行について計算
    titles = []
    embed_titles = []
    with open("titles.csv", "r") as f:
        for line in f:
            line = line.strip()
            input_ids= torch.tensor(tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0) 
            embed_l = torch.mean(model(input_ids, output_hidden_states=True).hidden_states[12], 1)
            titles.append(line)
            embed_titles.append(embed_l)

    # コサイン類似度の計算
    cos_sim = np.zeros(len(embed_titles))
    for j in range(len(embed_titles)):
        embed_j = embed_titles[j]
        dj = torch.cosine_similarity(embed_q, embed_j, dim=1)
        # 雑にtorchを処理しています（真似しちゃだめ）
        cos_sim[j] = dj[0]

    # 上位K個取ってくる (argpartition + argsort)
    # 参考: https://naoyashiga.hatenablog.com/entry/2017/04/13/224339
    K = 10
    unsorted_max_indices = np.argpartition(-cos_sim, K)[:K]
    y = cos_sim[unsorted_max_indices]
    indices = np.argsort(-y)
    max_k_indices = unsorted_max_indices[indices]
    for j in max_k_indices:
        print(f"{j} {cos_sim[j]:>.3f} {titles[j]}")