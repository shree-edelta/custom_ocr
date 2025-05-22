import string
import pickle

# Define character set (can include punctuation or special characters)
vocab = list(string.ascii_lowercase + string.ascii_uppercase + "0123456789")

# Assign unique integer to each character (0 for CTC blank label)
char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}  # 0 reserved
idx2char = {idx: char for char, idx in char2idx.items()}
print(char2idx)
# print(idx2char)
# Save tokenizer
with open("4token.pickle", "wb") as f:
    token = pickle.dump({"char2idx": char2idx, "idx2char": idx2char}, f)
    
print(type(token))
