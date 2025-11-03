wandb_log = True
wandb_project = 'reddit-user-comments'
wandb_run_name = 'gpt-reddit-conservative'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this results in ~1 epoch over the conservative Reddit user comments data (~2.25B tokens)
max_iters = 4570
lr_decay_iters = 4570

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

out_dir = 'out-reddit-conservative'
dataset = 'reddit_user_comments/conservative'