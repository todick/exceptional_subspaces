import argparse
import glob
import pandas as pd
import os

def concat(folder):
    casestudies = [d for d in os.listdir(folder) if os.path.isdir(f"{folder}/{d}") and not d == "tex"]

    for casestudy in casestudies:
        files = glob.glob(f"{folder}/{casestudy}/*.csv")
        files.sort()
        print(f"Found {len(files)} files in {casestudy}")
        
        dfs = [pd.read_csv(file) for file in files]
        df = pd.concat(dfs)
        output_file = f"{folder}/{casestudy}.csv"
        print(f"Writing {len(df)} rows to {output_file.split(os.sep)[-1]}")
        df.to_csv(output_file, index=False)

# return list of folders at exact depth
def find_folders_at_depth(path, depth):
    folders = []
    for root, dirs, files in os.walk(path):
        current_depth = root.count(os.sep) - path.count(os.sep)
        if current_depth == depth:
            folders.append(root)
    return folders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--depth', type=int, default=1)
    args = parser.parse_args()
    
    if args.depth == 1:
        concat(args.path)
    else:
        folders = find_folders_at_depth(args.path, args.depth)
        for folder in folders:
            print(f"Concatenating files in {folder}")
            concat(folder)