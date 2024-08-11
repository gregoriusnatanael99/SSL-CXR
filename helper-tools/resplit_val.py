import pickle
import os
import argparse
from datetime import datetime as dt
from tqdm import tqdm
import shutil

def check_val_folder(tgtFolder):
    if 'val' in os.listdir(tgtFolder):
        for classes in os.listdir(os.path.join(tgtFolder,'val')):
            print(f"Moving contents from {os.path.join(tgtFolder,'val',classes)} to {os.path.join(tgtFolder,'train',classes)} . . .")
            for i in tqdm(os.listdir(os.path.join(tgtFolder,'val',classes))):
                shutil.move(os.path.join(tgtFolder,'val',classes,i),os.path.join(tgtFolder,'train',classes,i))
        return True
    else: 
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--filename", type=str, default='val_imgs.pkl')

    args = parser.parse_args()

    if not check_val_folder(args.dir):
        print("Val folder does not exist!")
    else:
        print("Resplitting dataset . . .")
        with open(args.filename, 'rb') as fd:
            val_list = pickle.load(fd)

        for classes in os.listdir(os.path.join(args.dir,'train')):
            img_to_move = os.listdir(os.path.join(args.dir,'train',classes))
            ori_len = len(img_to_move)
            img_to_move = [i for i in img_to_move if i in val_list]
            print(f"Found {len(img_to_move)} files to move from {ori_len} files.")
            print(f"Moving contents from {os.path.join(args.dir,'train',classes)} to {os.path.join(args.dir,'val',classes)} . . .")
            
            for i in tqdm(img_to_move):
                shutil.move(os.path.join(args.dir,'train',classes,i),os.path.join(args.dir,'val',classes,i))
        
        print("Resplit finished")
