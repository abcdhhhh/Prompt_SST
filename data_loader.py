import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_df(fn: str, verbose=True):
    df = pd.read_csv(fn, delimiter='\t')
    if verbose:
        print("Passed %d entries from %s" % (len(df), fn))
    return df


def get_dataloader(df, tokenizer, prompt: bool, template: str, batch_size: int, test=False):
    input_ids = []
    attention_masks = []
    for sent in df.sentence:
        encoded_dict = tokenizer.encode_plus(
            (template % sent) if prompt else sent,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    print("input_ids size: ", input_ids.size())
    if test:
        dataset = TensorDataset(input_ids, attention_masks)
        return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
    else:
        labels = torch.tensor(df.label)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
