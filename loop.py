import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import pandas as pd


def train_loop(model, train_dl, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    with tqdm(iterable=train_dl) as pbar:
        for batch in pbar:
            pred = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device))
            y = batch[2].to(device)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": loss})
        avg_loss = total_loss / len(train_dl)
        print("avg_loss: ", avg_loss)
        return avg_loss


def dev_loop(model, dev_dl, device):
    model.eval()
    with tqdm(iterable=dev_dl) as pbar:
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        for batch in pbar:
            pred = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device)).argmax(dim=-1)
            y = batch[2].to(device)
            # print(pred, y)
            true_pos += torch.sum(pred & y).item()
            false_neg += torch.sum((pred == 0) & y).item()
            false_pos += torch.sum(pred & (y == 0)).item()
            true_neg += torch.sum((pred == 0) & (y == 0)).item()
            pbar.set_postfix({"(tp,fp,fn,tn)": (true_pos, false_pos, false_neg, true_neg)})
        true = true_pos + true_neg
        total = true + false_pos + false_neg
        acc = true / total
        eps = 1e-6
        prec = true_pos / (true_pos + false_pos + eps)
        rec = true_pos / (true_pos + false_neg + eps)
        f1 = 2 / (1 / (prec + eps) + 1 / (rec + eps))
        print("acc: ", acc, "prec: ", prec, "rec: ", rec, "f1: ", f1)
        return acc


def test_loop(model, dev_dl, fn, device):
    model.eval()
    df = []
    i = 0
    with tqdm(iterable=dev_dl) as pbar:
        for batch in pbar:
            pred = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device)).argmax(dim=-1).tolist()
            for p in pred:
                df.append([i, p])
                i += 1
    df = pd.DataFrame(df, columns=['index', 'prediction'])
    df.to_csv(fn, sep='\t', index=False)
