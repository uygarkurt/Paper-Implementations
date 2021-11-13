from train import train
from utils import make_batch
from model import NNLM
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    sentences = [
        "i like dog",
        "i love coffee",
        "i hate milk"
    ]

    EMBEDDING_DIM = 2
    HIDDEN = 2
    CONTEXT_SIZE = 2

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)

    model = NNLM(n_class, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch(word_dict, sentences)
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    model = train(sentences, input_batch, target_batch, model, optimizer, criterion)

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
