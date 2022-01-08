import os
num_samples = [32, 64, 128]
prompts = [False, True]
for num_sample in num_samples:
    for prompt in prompts:
        args = '-h trainer.py' + ' --PROMPT=' + str(prompt) + ' --NUM_SAMPLES=' + str(num_sample) + ' --SAVE=' + str(prompt) + ' --TEST=' + str(prompt)
        print(args)
        args = args.strip().split(' ')
        pid = os.fork()
        if pid == 0:
            os.execv('/opt/anaconda/bin/python3', args)
        elif pid > 0:
            os.waitpid(pid, 0)
        else:
            print('fork failed.')
            exit(0)
