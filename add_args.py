import argparse


def add_args():
    parser = argparse.ArgumentParser(description='hypers')
    add_arg = parser.add_argument
    add_arg('--MODEL', type=str, choices=['bert', 'albert', 'roberta'], default='bert')
    add_arg('--PROMPT', type=bool, default=True)
    add_arg('--TEMPLATE', type=str, choices=['manual', 'prefix', 'ptuning', 'ptr', 'mixed'], default='manual')
    add_arg('--VERBALIZER', type=str, choices=['manual', 'auto', 'knowledgeable', 'ptr', 'generation', 'soft'], default='manual')
    add_arg('--LEARNING_RATE', type=float, default=5e-4)
    add_arg('--WEIGHT_DECAY', type=float, default=1e-2)
    add_arg('--NUM_SAMPLES', type=int, choices=[0, 32, 64, 128, 8544], default=0)
    add_arg('--SAVE_MODEL', type=bool, default=False)
    args = parser.parse_args()
    return args
