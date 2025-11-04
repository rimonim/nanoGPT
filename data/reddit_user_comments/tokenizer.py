# train a tokenizer on subset of Reddit conservatives comment histories
# (individual Reddit user histories are saved as csv.gz files in `input_file_dir`)

import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, normalizers, decoders
from preprocess import preprocess_reddit_comments, list_files

input_file_dir = '/Volumes/Crucial X9/projects/cultural_depolarization/user_text/conservative'
output_tokenizer_path = os.path.join(os.path.dirname(__file__), 'conservative/meta.json')
# input_file_dir = '/Volumes/Crucial X9/projects/cultural_depolarization/user_text/liberal'
# output_tokenizer_path = os.path.join(os.path.dirname(__file__), 'liberal/meta.json')

texts = preprocess_reddit_comments(list_files(input_file_dir), sample_size=10000)
print(f'Loaded {len(texts)} user comment histories for tokenizer training')

# initialize a tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=32768, min_frequency=100, special_tokens=['<|endoftext|>', '<|url|>'],initial_alphabet=[chr(i) for i in range(256)])
tokenizer.train_from_iterator(texts, trainer=trainer)

# save the tokenizer
tokenizer.save(output_tokenizer_path)
print(f'Trained tokenizer saved to {output_tokenizer_path}')