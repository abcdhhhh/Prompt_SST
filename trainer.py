import torch
from torch import nn
from torch.optim import AdamW
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, PrefixTuningTemplate, PtuningTemplate, PTRTemplate, MixedTemplate
from openprompt.prompts import ManualVerbalizer, AutomaticVerbalizer, KnowledgeableVerbalizer, PTRVerbalizer, GenerationVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import fitlog
from add_args import add_args
from data_loader import get_dataset

args = add_args()

# fitlog
fitlog.set_log_dir('logs/')
fitlog.commit(__file__)  # auto commit your codes
fitlog.add_hyper_in_file(__file__)  # record your hyperparameters
fitlog.add_hyper(args)

# set device
device = 0 if torch.cuda.is_available() else 'cpu'

# prepare data
if args.NUM_SAMPLES == 0:
    train_data = []
elif args.NUM_SAMPLES == 32:
    train_data = get_dataset('FewShotSST/train_32.tsv')
elif args.NUM_SAMPLES == 64:
    train_data = get_dataset('FewShotSST/train_64.tsv')
elif args.NUM_SAMPLES == 128:
    train_data = get_dataset('FewShotSST/train_128.tsv')
elif args.NUM_SAMPLES == 8544:
    train_data = get_dataset('FewShotSST/train.tsv')
else:
    assert (0)
dev_data = get_dataset('FewShotSST/dev.tsv')
test_data = get_dataset('FewShotSST/test.tsv', test=True)

# set plm
if args.MODEL == 'bert':
    plm, tokenizer, plm_config, tokenizer_wrapper_class = load_plm("bert", "bert-base-cased")
elif args.MODEL == 'albert':
    plm, tokenizer, plm_config, tokenizer_wrapper_class = load_plm("albert", "albert-base-v2")
elif args.MODEL == 'roberta':
    plm, tokenizer, plm_config, tokenizer_wrapper_class = load_plm("roberta", "roberta-base")
else:
    assert (0)
for name, param in plm.named_parameters():
    if name.startswith(args.MODEL):
        param.requires_grad = False

# set template
if args.TEMPLATE == 'manual':
    template = ManualTemplate(text='{"placeholder":"text_a"} It was {"mask"}', tokenizer=tokenizer)
elif args.TEMPLATE == 'prefix':
    template = PrefixTuningTemplate(model=plm, plm_config=plm_config, tokenizer=tokenizer)
elif args.TEMPLATE == 'ptuning':
    template = PtuningTemplate(model=plm, tokenizer=tokenizer)
elif args.TEMPLATE == 'ptr':
    template = PTRTemplate(model=plm, tokenizer=tokenizer)
elif args.TEMPLATE == 'mixed':
    template = MixedTemplate(model=plm, tokenizer=tokenizer)

# set verbalizer
classes = ["negative", "positive"]
if args.VERBALIZER == 'manual':
    verbalizer = ManualVerbalizer(tokenizer=tokenizer, classes=classes, label_words={"negative": ["bad"], "positive": ["good", "wonderful", "great"]})
elif args.VERBALIZER == 'auto':
    verbalizer = AutomaticVerbalizer(tokenizer=tokenizer, classes=classes)
elif args.VERBALIZER == 'knowledgeable':
    verbalizer = KnowledgeableVerbalizer(tokenizer=tokenizer, classes=classes)
elif args.VERBALIZER == 'ptr':
    verbalizer = PTRVerbalizer(tokenizer=tokenizer, classes=classes)
elif args.VERBALIZER == 'generation':
    verbalizer = GenerationVerbalizer(tokenizer=tokenizer, classes=classes)
elif args.VERBALIZER == 'soft':
    verbalizer = SoftVerbalizer(tokenizer=tokenizer, classes=classes)

# set model
model = PromptForClassification(template=template, plm=plm, verbalizer=verbalizer)
model = model.to(device)

# set data_loader
if args.NUM_SAMPLES == 0:
    print("No training data")
else:
    train_dl = PromptDataLoader(dataset=train_data, tokenizer=tokenizer, template=template, tokenizer_wrapper_class=tokenizer_wrapper_class)
dev_dl = PromptDataLoader(dataset=dev_data, tokenizer=tokenizer, template=template, tokenizer_wrapper_class=tokenizer_wrapper_class)
test_dl = PromptDataLoader(dataset=test_data, tokenizer=tokenizer, template=template, tokenizer_wrapper_class=tokenizer_wrapper_class)
print("data loaders set")

# set optimizer
optimizer = AdamW(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)

# set loss
loss_fn = nn.CrossEntropyLoss()

# train
print("TRAIN!")
if args.NUM_SAMPLES == 0:
    print("No training needed")
else:
    for batch in train_dl:
        pred = model(batch)
        y = batch["label"]
        loss = loss_fn(pred, y)
        loss.backward()

if args.SAVE_MODEL == 'True':
    print("SAVE!")

# fail to use fitlog
fitlog.finish()
