import codecs
from openprompt.data_utils import InputExample


def get_dataset(fn: str, verbose=True, test=False):
    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    dataset = []
    for i, line in enumerate(lines[1:]):
        if test:
            _, words = line.split('\t')
            dataset.append(InputExample(guid=i, text_a=words))
        else:
            words, target = line.split('\t')
            dataset.append(InputExample(guid=i, text_a=words, label=int(target)))
    if verbose:
        print('Loading from %s: %d entries' % (fn, len(dataset)))
    return dataset
