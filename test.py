from transformers import AlbertForMaskedLM, AlbertTokenizer
import torch

model = AlbertForMaskedLM.from_pretrained('albert-base-v2')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

sentence = "It is a very beautiful book."
tokens = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
# tokens = tokenizer.tokenize(sentence)
# i就是被mask掉的id
for i in range(1, len(tokens) - 1):
    tmp = tokens[:i] + ['[MASK]'] + tokens[i + 1:]
    masked_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tmp)])
    segment_ids = torch.tensor([[0] * len(tmp)])

    outputs = model(masked_ids, token_type_ids=segment_ids)
    prediction_scores = outputs[0]
    print(tmp)
    print(prediction_scores.size())
    # 打印被预测的字符
    prediction_index = torch.argmax(prediction_scores[0, i]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([prediction_index])[0]
    print(predicted_token)
