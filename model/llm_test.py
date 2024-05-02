import torch
import llm_model

from llm_model import AttentionModule, FeedforwardNetwork, MultiHeadModule, TransformerBlock, LLMModel


DEVICE = llm_model.DEVICE
model = torch.load("../pretrained_model/LLM.pth")
model.eval()

start = model.word_to_index['in']
x = torch.tensor([[start]], dtype=torch.long, device=DEVICE)
print(model.generate(x))


