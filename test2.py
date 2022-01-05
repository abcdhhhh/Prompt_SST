from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader

classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative", "positive"
]
dataset = [  # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(guid=1, text_a="Albert Einstein was one of the greatest intellects of his time.", label=0),
    InputExample(guid=2, text_a="The film was badly made.", label=1),
]

plm, tokenizer, model_config, WrapperClass = load_plm("albert", "albert-base-v2")

promptTemplate = ManualTemplate(
    text='{"placeholder":"text_a"} It was {"mask"}',
    tokenizer=tokenizer,
)
promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer=tokenizer,
)

model = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)
data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)
for batch in data_loader:
    print(batch["label"])
    print(model(batch))
