This project is based on BERT. (https://huggingface.co/docs/transformers/model_doc/bert)

Please install the packages before training.

```bash
pip install -r requirements.txt
```

The dataset is `./FewShotSST/` .

And if you run the command

```bash
python trainer.py --NUM_SAMPLES=32 --SAVE=True --TEST=True
```

The predictions of `./FewShotSST/test.tsv` will be saved in `./output/32.tsv` .

Must save before test.

