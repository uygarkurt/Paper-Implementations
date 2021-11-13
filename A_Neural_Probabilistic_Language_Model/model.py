import torch
import torch.nn.functional as F
import torch.nn as nn

class NNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(NNLM, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size, bias = False)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1,self.context_size * self.embedding_dim))
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
