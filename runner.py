import os
num_samples = [32, 64, 128]
prompts = [False, True]
for num_sample in num_samples:
    for prompt in prompts:
        args = '-h trainer.py' + ' --PROMPT=' + str(prompt) + ' --NUM_SAMPLES=' + str(num_sample)
        args = args.strip().split(' ')
        os.execv('/opt/anaconda/bin/python3', args)
