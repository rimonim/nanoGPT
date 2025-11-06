wandb_log = True
wandb_project = 'reddit-user-comments'
wandb_run_name = 'gpt-reddit-conservative'

n_layer = 16
n_head = 16
n_embd = 1024

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 1 GPUs = 61,440
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 1

# this results in ~4 epochs over the conservative Reddit user comments data (~9B tokens)
max_iters = 36015*4
lr_decay_iters = 36015*4

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

out_dir = 'out-reddit-conservative'
dataset = 'reddit_user_comments/conservative'