# reformats quanteda tokens object to a binary file for training
library(quanteda)
original_wd <- getwd()
setwd("~/Projects/cognitive_embeddings/")
source("load_coca.R")

coca_tokens <- unclass(coca_tokens)

set.seed(1234)
test_docs <- sample(seq_along(coca_tokens), length(coca_tokens)/400) # 0.25% for testing

# 1. Extract the token IDs as a flat integer vector
train_tokens <- unlist(coca_tokens[-test_docs], use.names = FALSE) - 1L # zero-indexed
test_tokens <- unlist(coca_tokens[test_docs], use.names = FALSE) - 1L # zero-indexed

# 2. Write to binary file in uint16 format
setwd(original_wd)
con <- file("data/coca/train.bin", "wb")
writeBin(as.integer(train_tokens), con, size = 2, endian = "little")
close(con)

con <- file("data/coca/test.bin", "wb")
writeBin(as.integer(test_tokens), con, size = 2, endian = "little")
close(con)

# Construct a meta.pkl file with vocab mappings
# meta['stoi'] = string to index dict, meta['itos'] = index to string dict
library(reticulate)
meta <- list()
vocab <- attr(coca_tokens, "types")
stoi <- as.list(setNames(seq_along(vocab) - 1L, vocab))
itos <- as.list(setNames(vocab, seq_along(vocab) - 1L))
meta$stoi <- stoi
meta$itos <- itos
py_save_object(meta, "data/coca/meta.pkl")