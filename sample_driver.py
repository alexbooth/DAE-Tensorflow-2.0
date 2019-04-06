import subprocess

run_num = 292
epochs = 1000
for i in range(run_num, epochs):
    cmd = "python3 train.py --epochs=1 --batch_size=256 --keep_training".split()
    p = subprocess.Popen(cmd)
    p.communicate()

    cmd = "python3 sample.py --run_num {}".format(i).split()
    p = subprocess.Popen(cmd)
    p.communicate()
    print("Finished iteration {}".format(i))
