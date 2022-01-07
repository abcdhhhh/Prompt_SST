import argparse


def add_args():
    parser = argparse.ArgumentParser(description='hypers')
    add_arg = parser.add_argument
    add_arg('--N_EPOCHS', type=int, default=4)
    add_arg('--BATCH_SIZE', type=int, default=8)
    add_arg('--MODEL', type=str, choices=['bert', 'albert', 'roberta'], default='bert')
    add_arg('--PROMPT', type=bool, default=True)
    add_arg('--TEMPLATE', type=str, default='%s An [MASK] film.')
    add_arg('--V_NEG', type=str, default='awful')
    add_arg('--V_POS', type=str, default='enjoyable')
    add_arg('--LEARNING_RATE', type=float, default=2e-5)
    add_arg('--NUM_SAMPLES', type=int, choices=[0, 32, 64, 128, 8544], default=0)
    add_arg('--SAVE', type=bool, default=False)
    args = parser.parse_args()
    return args
