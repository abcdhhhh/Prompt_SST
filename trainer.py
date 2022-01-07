import torch
from torch import nn
from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer
from transformers import BertForMaskedLM, AlbertForMaskedLM, RobertaForMaskedLM
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, RobertaForSequenceClassification
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

# set tokenizer
print('loading tokenizer ...')
if args.MODEL == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
elif args.MODEL == 'albert':
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
elif args.MODEL == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
else:
    assert (0)
sentence = test_df.sentence[0]
print("tokenizer loaded")
mask_id, p_neg, p_pos = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' [MASK] ' + args.V_NEG + ' ' + args.V_POS))
print("mask_id, p_neg, p_pos: ", mask_id, p_neg, p_pos)

# prepare dl
print("preparing dataloader")
if args.NUM_SAMPLES == 0:
    print("No training data")
else:
    train_dataloader = get_dataloader(train_df, tokenizer, args.PROMPT, args.TEMPLATE, batch_size=args.BATCH_SIZE, train=True)
dev_dataloader = get_dataloader(dev_df, tokenizer, args.PROMPT, args.TEMPLATE, batch_size=32)
test_dataloader = get_dataloader(test_df, tokenizer, args.PROMPT, args.TEMPLATE, batch_size=32, test=True)
print("dataloader prepared")

# set model
print("loading model ...")
if args.PROMPT:
    if args.MODEL == 'bert':
        model = BertForMaskedLM.from_pretrained("bert-large-uncased")
    elif args.MODEL == 'albert':
        model = AlbertForMaskedLM.from_pretrained("albert-base-v2")
    elif args.MODEL == 'roberta':
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
    model = BertPrompt(model, p_neg, p_pos, mask_id)
else:
    if args.MODEL == 'bert':
        model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2, output_attentions=False, output_hidden_states=False)
    elif args.MODEL == 'albert':
        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2, output_attentions=False, output_hidden_states=False)
    elif args.MODEL == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2, output_attentions=False, output_hidden_states=False)
    model = Bert(model)
# model = model.to(device)
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
if args.NUM_SAMPLES == 0:
    print("No training needed")
    print("first validation ...")
    # dev_loop(model, dev_dataloader)
else:
    for epoch in range(args.N_EPOCHS):
        print("EPOCH: ", epoch)
        train_loop(model, train_dataloader, loss_fn, optimizer, scheduler)
        dev_loop(model, dev_dataloader)
print("training completed")

# save
if args.SAVE:
    print("SAVE!")
    test_loop(model, test_dataloader, fn='FewShotSST/' + str(args.NUM_SAMPLES) + '.tsv')
