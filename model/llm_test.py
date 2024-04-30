import torch
import LLM_own_embedding

from LLM_own_embedding import AttentionModule, FeedforwardNetwork, MultiHeadModule, TransformerBlock, LLMModel


DEVICE = LLM_own_embedding.DEVICE
model = torch.load("LLM.pth")
model.eval()

start = model.word_to_index['in']LLMModel
x = torch.tensor([[start]], dtype=torch.long, device=DEVICE)
print(model.generate(x))

