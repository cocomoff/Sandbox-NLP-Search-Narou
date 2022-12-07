import sentencepiece
from transformers import AlbertTokenizer, AlbertForPreTraining
import torch


if __name__ == '__main__':
    query = '悪徳令嬢'
    tokenizer = AlbertTokenizer.from_pretrained('ALINEAR/albert-japanese-v2')
    model = AlbertForPreTraining.from_pretrained('ALINEAR/albert-japanese-v2')

    input_ids= torch.tensor(tokenizer.encode(query, add_special_tokens=True)).unsqueeze(0) 
    embeddings = torch.mean(model(input_ids, output_hidden_states=True).hidden_states[12], 1)
    print(embeddings.shape)
    print(embeddings[0, :10])