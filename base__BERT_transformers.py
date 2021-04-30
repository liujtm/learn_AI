import torch
from transformers import BertModel, BertTokenizer
# 这里我们调用bert-base模型，同时模型的词典经过小写处理
model_name = 'bert-base-uncased'
# 读取模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
# 载入模型
model = BertModel.from_pretrained(model_name)
# 输入文本
input_text = "Here is some text to encode"
# 通过tokenizer把文本变成 token_id
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
print(tokenizer.tokenize(text=input_text))
print(tokenizer.convert_ids_to_tokens(input_ids))
# input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
input_batch = torch.tensor([input_ids])  # 注意 [input_ids]，多加一层维度，batch size为1
# 获得BERT模型最后一个隐层结果
with torch.no_grad():
    last_hidden_states = model(input_batch)[0]  # Models outputs are now tuples

""" tensor([[[-0.0549,  0.1053, -0.1065,  ..., -0.3550,  0.0686,  0.6506],
         [-0.5759, -0.3650, -0.1383,  ..., -0.6782,  0.2092, -0.1639],
         [-0.1641, -0.5597,  0.0150,  ..., -0.1603, -0.1346,  0.6216],
         ...,
         [ 0.2448,  0.1254,  0.1587,  ..., -0.2749, -0.1163,  0.8809],
         [ 0.0481,  0.4950, -0.2827,  ..., -0.6097, -0.1212,  0.2527],
         [ 0.9046,  0.2137, -0.5897,  ...,  0.3040, -0.6172, -0.1950]]]) 
	shape: (1, 9, 768)       (batch_size, 本句中词语数目, hidden_nodes)
"""
