import torch
from torch import nn
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
from transformers import BertForMaskedLM, DistilBertForMaskedLM, RobertaForMaskedLM
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, RobertaForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from add_args import add_args
from data_loader import get_df, get_dataloader
from models import Bert, BertPrompt
from loop import train_loop, dev_loop, test_loop

args = add_args()

# set device
device = 0 if torch.cuda.is_available() else 'cpu'
print("device: ", device)

# prepare df
print("preparing df")
if args.NUM_SAMPLES == 0:
    print("No training data")
elif args.NUM_SAMPLES == 32:
    train_df = get_df('FewShotSST/train_32.tsv')
elif args.NUM_SAMPLES == 64:
    train_df = get_df('FewShotSST/train_64.tsv')
elif args.NUM_SAMPLES == 128:
    train_df = get_df('FewShotSST/train_128.tsv')
elif args.NUM_SAMPLES == 8544:
    train_df = get_df('FewShotSST/train.tsv')
else:
    assert (0)
dev_df = get_df('FewShotSST/dev.tsv')
test_df = get_df('FewShotSST/test.tsv')
print("df prepared")

# set pretrain
pretrain = {'bert': 'bert-large-uncased', 'distilbert': 'distilbert-base-uncased', 'roberta': 'roberta-large'}

# set tokenizer
print('loading tokenizer ...')
if args.MODEL == 'bert':
    tokenizer = BertTokenizer.from_pretrained(pretrain[args.MODEL])
elif args.MODEL == 'distilbert':
    tokenizer = DistilBertTokenizer.from_pretrained(pretrain[args.MODEL])
elif args.MODEL == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained(pretrain[args.MODEL])
else:
    assert (0)
sentence = test_df.sentence[0]
print("tokenizer loaded")
if args.MODEL in ['bert', 'distilbert']:
    mask = '[MASK]'
elif args.MODEL in ['roberta']:
    mask = '<mask>'
mask_id = tokenizer.convert_tokens_to_ids(mask)
p_neg = tokenizer.convert_tokens_to_ids(args.V_NEG)
p_pos = tokenizer.convert_tokens_to_ids(args.V_POS)
print("mask_id, p_neg, p_pos: ", mask_id, p_neg, p_pos)

# prepare dl
print("preparing dataloader")
if args.NUM_SAMPLES == 0:
    print("No training data")
else:
    train_dataloader = get_dataloader(train_df, tokenizer, args.PROMPT, args.TEMPLATE % ('%s', mask), batch_size=args.BATCH_SIZE, train=True)
dev_dataloader = get_dataloader(dev_df, tokenizer, args.PROMPT, args.TEMPLATE % ('%s', mask), batch_size=args.BATCH_SIZE)
test_dataloader = get_dataloader(test_df, tokenizer, args.PROMPT, args.TEMPLATE % ('%s', mask), batch_size=args.BATCH_SIZE, test=True)
print("dataloader prepared")

# set model
print("loading model ...")
if args.PROMPT:
    if args.MODEL == 'bert':
        model = BertForMaskedLM.from_pretrained(pretrain[args.MODEL])
    elif args.MODEL == 'distilbert':
        model = DistilBertForMaskedLM.from_pretrained(pretrain[args.MODEL])
    elif args.MODEL == 'roberta':
        model = RobertaForMaskedLM.from_pretrained(pretrain[args.MODEL])
    model = BertPrompt(model, p_neg, p_pos, mask_id)
else:
    if args.MODEL == 'bert':
        model = BertForSequenceClassification.from_pretrained(pretrain[args.MODEL], num_labels=2, output_attentions=False, output_hidden_states=False)
    elif args.MODEL == 'distilbert':
        model = DistilBertForSequenceClassification.from_pretrained(pretrain[args.MODEL], num_labels=2, output_attentions=False, output_hidden_states=False)
    elif args.MODEL == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(pretrain[args.MODEL], num_labels=2, output_attentions=False, output_hidden_states=False)
    model = Bert(model)
model = model.to(device)
print("model loaded")

# set optimizer
print("setting optimizer")
if args.NUM_SAMPLES == 0:
    print("no training needed")
else:
    optimizer = AdamW(model.parameters(), lr=args.LEARNING_RATE)
    total_steps = len(train_dataloader) * args.N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()
    print("optimizer set")

# train
print("start training ...")
if args.SAVE:
    fn = "model/" + str(args.NUM_SAMPLES) + str(args.PROMPT) + ".pt"
    print("save dir: ", fn)
if args.NUM_SAMPLES == 0:
    print("No training needed")
    print("first validation ...")
    dev_loop(model, dev_dataloader, device)
    if args.SAVE:
        torch.save(model, fn)
        print("model saved")
else:
    best_acc = 0
    es = 0
    for epoch in range(args.N_EPOCHS):
        print("EPOCH: ", epoch)
        train_loop(model, train_dataloader, loss_fn, optimizer, scheduler, device)
        val_acc = dev_loop(model, dev_dataloader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            es = 0
            if args.SAVE:
                torch.save(model, fn)
                print("model saved")
        else:
            es += 1
            if es > 2:
                print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", val_acc)
                break

    print("training completed")

if args.TEST:
    print("TEST!")
    assert (args.SAVE)
    model = torch.load(fn)
    model = model.to(device)
    test_loop(model, test_dataloader, fn='output/' + str(args.NUM_SAMPLES) + '.tsv', device=device)
    print("test end")
