import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import pandas as pd


def train_loop(model, train_dl, loss_fn, optimizer, scheduler):
    model.train()
    total_loss = 0
    with tqdm(iterable=train_dl) as pbar:
        for batch in pbar:
            pred = model(input_ids=batch[0], attention_mask=batch[1])
            y = batch[2]
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": loss})
        print("avg_loss: ", total_loss / len(train_dl))


def dev_loop(model, dev_dl):
    model.eval()
    with tqdm(iterable=dev_dl) as pbar:
        total_true = 0
        total_len = 0
        for batch in pbar:
            pred = model(input_ids=batch[0], attention_mask=batch[1]).argmax(dim=-1)
            y = batch[2]
            # print(pred, y)
            total_true += torch.sum(pred == y).item()
            total_len += len(y)
            pbar.set_postfix({"true": total_true, "len": total_len})
        print("total_acc: ", total_true / total_len)


def test_loop(model, dev_dl, fn):
    model.eval()
    df = []
    i = 0
    with tqdm(iterable=dev_dl) as pbar:
        for batch in pbar:
            pred = model(input_ids=batch[0], attention_mask=batch[1]).argmax(dim=-1).tolist()
            for p in pred:
                df.append([i, pred])
                i += 1
    df = pd.DataFrame(df, columns=['index', 'prediction'])
    df.to_csv(fn)
