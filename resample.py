import os
import shutil
from datetime import datetime as dt
import argparse
import random
random.seed(12321)
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--rate", type=float, default=0.5)

    args = parser.parse_args()

    start = dt.now()
    for subfolder in os.listdir(args.dir):
        filenames = os.listdir(os.path.join(args.dir,subfolder))
        samples = random.sample(filenames,int(args.rate*len(filenames)))
        print(f'Processing folder {subfolder} . . .')
        os.makedirs(os.path.join(args.dest,subfolder),exist_ok=True)
        for i in tqdm(samples):
            shutil.move(src=os.path.join(args.dir,subfolder,i),dst=os.path.join(args.dest,subfolder,i))
    end = dt.now()
    print(f"Process finished in {end-start}")