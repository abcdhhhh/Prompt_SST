import torch
from tqdm import tqdm


def train_loop(model, train_dl, loss_fn, optimizer):
    model.train()
    total_loss = 0
    with tqdm(iterable=train_dl) as pbar:
        for batch in pbar:
            pred = model(batch)
            y = batch["label"]
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss})
        print("avg_loss: ", total_loss / len(train_dl))


def dev_loop(model, dev_dl):
    model.eval()
    with tqdm(iterable=dev_dl) as pbar:
        total_true = 0
        total_len = 0
        for batch in pbar:
            pred = model(batch).argmax(dim=-1)
            y = batch["label"]
            total_true += torch.sum(pred == y).item()
            total_len += len(y)
            pbar.set_postfix({"true": total_true, "len": total_len})
        print("total_acc: ", total_true / total_len)
