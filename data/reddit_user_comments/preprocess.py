# Preprocess Reddit user comment histories for tokenizer training

import os
import glob
import pandas as pd
import re
import random
from tqdm import tqdm

def list_files(input_file_dir):
    return glob.glob(os.path.join(input_file_dir, '*.csv.gz'))

def preprocess_reddit_comments(files, idx=None, sample_size=None, progress_bar=True):
    # include only specific files if idx is provided
    if idx is not None:
        files = [files[i] for i in idx]
    
    # select sample_size files randomly
    if sample_size is not None:
        random.seed(41)
        files = random.sample(files, sample_size)

    # load sample of user comment histories
    # each `comment` should be preceded by subreddit + datetime and each user history preceded by <|endoftext|>
    texts = []
    skipped = 0
    for file in tqdm(files, desc='Preprocessing Reddit comment histories', disable=not progress_bar):
        try:
            df = pd.read_csv(file, compression='infer', encoding='utf-8')
        except Exception as e:
            skipped += 1
            continue
        user_text = ''
        for _, row in df.iterrows():
            subreddit = row['subreddit']
            # use either created_utc or date_utc, depending on which is available
            date_utc = pd.to_datetime(row['created_utc'], unit='s').strftime('%Y-%m-%d %H:%M') if 'created_utc' in row else row['date_utc']
            comment = row['body'] if 'body' in row else row['comment']
            user_text += f'[{subreddit} - {date_utc}] {comment}\n'
        texts.append(f'<|endoftext|>\n{user_text}\n')

    if skipped > 0:
        print(f'Skipped {skipped} files due to read errors')
    
    # replace full urls with special token
    texts = [re.sub(r"(?<=\()https?:\S*(?=\))", '<|url|>', text) for text in texts] # within parentheses
    texts = [re.sub(r"https?:\S*", '<|url|>', text) for text in texts] # standalone

    return texts