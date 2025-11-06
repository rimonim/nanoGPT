
## Reddit user comments dataset

Full comment histories of contributors to the most right-wing groups (conservative) and the most left-wing groups (liberal), maintaining each user history in chronological order. A subreddit name and timestamp precede each comment, but commenter usernames are omitted. 

After running `prepare.py` (preprocess) for each group we get:

- conservative/train.bin is ~4.2GB (53,102 users; 2,112,936,727 tokens)
- conservative/val.bin is ~22MB (267 users; 10,868,900 tokens)

- liberal/train.bin is ~7.9GB (83,899 users; 3,969,122,041 tokens)
- liberal/val.bin is ~38MB (422 users; 18,730,019 tokens)