# saves the reddit_conservatives dataset to a binary file for training

import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
from preprocess import preprocess_reddit_comments, list_files
import random

test_size = 0.005 # 0.5% of data for test split
batch_size = 200 # number of user histories to process in a batch

group = 'liberal'

input_file_dir = f'/Volumes/Crucial X9/projects/cultural_depolarization/user_text/{group}'
tokenizer_path = os.path.join(os.path.dirname(__file__), f'{group}/meta.json')
output_train_path = os.path.join(os.path.dirname(__file__), f'{group}/train.bin')
output_test_path = os.path.join(os.path.dirname(__file__), f'{group}/val.bin')

# load the trained tokenizer
tokenizer = Tokenizer.from_file(tokenizer_path)
print(f'Loaded tokenizer from {tokenizer_path}')
print(f'Tokenizer vocab size: {tokenizer.get_vocab_size()}')

if __name__ == '__main__':
    # preprocess reddit comments
    files = list_files(input_file_dir)

    # split texts randomly into train and test sets
    random.seed(2398)
    np.random.shuffle(files)
    split_idx = int(len(files) * (1 - test_size))
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    print(f'Train set: {len(train_files)} user comment histories')
    print(f'Test set: {len(test_files)} user comment histories')

     # tokenize and write train set incrementally
    train_token_count = 0
    with open(output_train_path, 'wb') as f:
        for batch_start in tqdm(range(0, len(train_files), batch_size), desc='Tokenizing train set'):
            batch_end = min(batch_start + batch_size, len(train_files))
            batch_files = train_files[batch_start:batch_end]
            texts = preprocess_reddit_comments(batch_files, progress_bar=False)
            
            batch_ids = []
            for text in texts:
                encoded = tokenizer.encode(text)
                batch_ids.extend(encoded.ids)
            
            # Write batch to file
            batch_arr = np.array(batch_ids, dtype=np.uint16)
            batch_arr.tofile(f)
            train_token_count += len(batch_ids)
    print(f'Train set: {train_token_count} tokens')

    # tokenize and write test set incrementally
    test_token_count = 0
    with open(output_test_path, 'wb') as f:
        for batch_start in tqdm(range(0, len(test_files), batch_size), desc='Tokenizing test set'):
            batch_end = min(batch_start + batch_size, len(test_files))
            batch_files = test_files[batch_start:batch_end]
            texts = preprocess_reddit_comments(batch_files, progress_bar=False)
            
            batch_ids = []
            for text in texts:
                encoded = tokenizer.encode(text)
                batch_ids.extend(encoded.ids)
            
            # Write batch to file
            batch_arr = np.array(batch_ids, dtype=np.uint16)
            batch_arr.tofile(f)
            test_token_count += len(batch_ids)
    print(f'Test set: {test_token_count} tokens')

    print(f'Saved train.bin with {train_token_count} tokens to {output_train_path}')
    print(f'Saved test.bin with {test_token_count} tokens to {output_test_path}')