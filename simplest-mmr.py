import sentencepiece
from transformers import AlbertTokenizer, AlbertForPreTraining
import numpy as np
import torch
import argparse

args = argparse.ArgumentParser()
args.add_argument("lbd", default=0.5, type=float)
input_args = args.parse_args()

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
    N = len(embed_titles)

    # コサイン類似度の計算
    cos_sim = np.zeros(N)
    for j in range(N):
        embed_j = embed_titles[j]
        dj = torch.cosine_similarity(embed_q, embed_j, dim=1)
        # 雑にtorchを処理しています（真似しちゃだめ）
        cos_sim[j] = dj[0]

    # 文書間のコサイン類似度の計算
    cos_sim_mat = np.zeros((N, N))
    for i in range(N):
        embed_i = embed_titles[i]
        for j in range(i + 1, N):
            embed_j = embed_titles[j]
            dij = torch.cosine_similarity(embed_i, embed_j, dim=1)
            cos_sim_mat[i, j] = cos_sim_mat[j, i] = dij[0]

    # MMR
    K = 10
    param_lambda = input_args.lbd
    results = []

    # 1アイテム目: 最大値のもの
    k = np.argmax(cos_sim)
    results.append(k)


    # 2アイテム目 ~ K=10アイテム目
    for i in range(1, K):
        new_sims = np.zeros_like(cos_sim)

        # 既に選択したアイテムは選択しないので、-infにしておく
        new_sims = param_lambda * cos_sim
        new_sims[results] = -np.float64("inf")

        # 雑な実装: 各アイテム D_i について、これまで選んだアイテムとのsimで最大値を
        # (1 - λ) を付けて引いて調整する
        for j in range(N):
            # アイテム j とこれまでに選んだアイテム (results) との類似度の最大値
            values = [cos_sim_mat[j, s] for s in results]
            max_j = max(values)
            new_sims[j] -= (1 - param_lambda) * max_j

        # スコアを調整した上で選ぶ
        k = np.argmax(new_sims)
        results.append(k)

    # 結果
    for j in results:
        print(f"{j} {cos_sim[j]:>.3f} {titles[j]}")