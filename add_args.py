import argparse


def add_args():
    parser = argparse.ArgumentParser(description='hypers')
    add_arg = parser.add_argument
    add_arg('--N_EPOCHS', type=int, default=2)
    add_arg('--BATCH_SIZE', type=int, default=32)
    add_arg('--MODEL', type=str, choices=['bert', 'albert', 'roberta'], default='albert')
    add_arg('--PROMPT', type=bool, default=True)
    add_arg('--TEMPLATE', type=str, choices=['%s It was [MASK].', '%s A [MASK] film.'], default='%s It was [MASK].')
    add_arg('--V_NEG', type=str, choices=['bad', 'terrible'], default='terrible')
    add_arg('--V_POS', type=str, choices=['good', 'great'], default='great')
    add_arg('--LEARNING_RATE', type=float, default=2e-5)
    add_arg('--NUM_SAMPLES', type=int, choices=[0, 32, 64, 128, 8544], default=0)
    add_arg('--SAVE_MODEL', type=bool, default=False)
    args = parser.parse_args()
    return args
