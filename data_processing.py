import os
import lzma
from tqdm import tqdm

# Function to list all .xz files in a directory
def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xz') and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

# file variables - saved in different location due to large size
folder_path = "E:/Saathvik/openwebtext"
output_file_train = "E:/Saathvik/openwebtext/train_split.txt"
output_file_test = "E:/Saathvik/openwebtext/test_split.txt"
vocab_file = "E:/Saathvik/openwebtext/vocab.txt"
files = xz_files_in_dir(folder_path)
total_files = len(files)

# Split files into train and test sets
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_test = files[split_index:]

vocab = set()

# Training Files
with open(output_file_train, 'w', encoding='utf-8') as train_output:
    for filename in tqdm(files_train, total = len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as file:
            text = file.read()
            train_output.write(text)
            chars = set(text)
            vocab.update(chars)

# Testing Files
with open(output_file_test, 'w', encoding='utf-8') as test_output:
    for filename in tqdm(files_test, total = len(files_test)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as file:
            text = file.read()
            test_output.write(text)
            chars = set(text)
            vocab.update(chars)

# Write the vocabulary to a file   
with open(vocab_file, 'w', encoding='utf-8') as vocab_output:
    for char in vocab:
        vocab_output.write(char + '\n')