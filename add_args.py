import argparse


def add_args():
    parser = argparse.ArgumentParser(description='hypers')
    add_arg = parser.add_argument
    add_arg('--N_EPOCHS', type=int, default=16)
    add_arg('--BATCH_SIZE', type=int, default=8)
    add_arg('--MODEL', type=str, choices=['bert', 'distilbert'], default='bert')
    add_arg('--PROMPT', type=bool, default=True)
    add_arg('--TEMPLATE', type=str, default="%s It's really [MASK].")
    add_arg('--V_NEG', type=str, default='awful')
    add_arg('--V_POS', type=str, default='excellent')
    add_arg('--LEARNING_RATE', type=float, default=2e-5)
    add_arg('--NUM_SAMPLES', type=int, choices=[0, 32, 64, 128, 8544], default=0)
    add_arg('--SAVE', type=bool, default=False)
    add_arg('--TEST', type=bool, default=False)
    args = parser.parse_args()
    return args
