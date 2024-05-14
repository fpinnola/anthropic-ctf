import random
import string

import numpy as np
import matplotlib.pyplot as plt
import torch



charset = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(charset)}
INDEX_TO_CHAR = {idx: char for char, idx in CHAR_TO_INDEX.items()}
transformations = np.zeros((32, 27, 27), dtype=int)
data_out = [{} for _ in range (32)]
vocab = " " + string.ascii_lowercase
seq_length = 32  

model_path = './model.pkl'
model = torch.load(model_path)


def update_transformations(input_text, output_text):
    """Update the transformation matrix based on input and output texts."""
    for i, (inp, out) in enumerate(zip(input_text, output_text[0])):
        inp_idx = CHAR_TO_INDEX[inp]
        out_idx = CHAR_TO_INDEX[out]
        transformations[i, inp_idx, out_idx] += 1

def tensor_to_text(tensor, vocab):
    """ Convert a batch of tensors back to text """
    texts = []
    max_indices = tensor.argmax(dim=2)
    for indices in max_indices:
        text = ''.join(vocab[idx] for idx in indices)
        texts.append(text)
    return texts

def text_to_tensor(text, vocab):
    """ Convert text to a one-hot tensor """
    vocab_index = {char: idx for idx, char in enumerate(vocab)}
    tensor = torch.zeros(len(text), len(vocab), dtype=torch.float32)
    for i, char in enumerate(text):
        if char in vocab_index:
            tensor[i, vocab_index[char]] = 1.0
    return tensor

def create_batch(texts, vocab, seq_length):
    """Create a batch of one-hot encoded texts."""
    batch = torch.zeros(len(texts), seq_length, len(vocab), dtype=torch.float32)
    for i, text in enumerate(texts):
        tensor = text_to_tensor(text.ljust(seq_length)[:seq_length], vocab)  # Pad or truncate
        batch[i] = tensor
    return batch.view(len(texts), -1)  # Reshape to (batch_size, seq_length * vocab_size)

def generate_random_strings(size, length, existing_set):
    """ Generate a set of unique random strings. """
    chars = string.ascii_lowercase + " "  # Include space in the charset
    new_strings = set()

    prefix = ''

    while len(new_strings) < size:
        s = prefix + ''.join(random.choice(chars) for _ in range(length - len(prefix)))
        if s not in existing_set:  # Ensure uniqueness
            new_strings.add(s)
            existing_set.add(s)

    return list(new_strings)


generated_strings = set()

for _ in range(10000):
        # Generate random strings
        random_strings = generate_random_strings(64, 32, generated_strings)
        # Tensor to store batch
        batch_size = len(random_strings)
        batch_tensor = create_batch(random_strings, vocab, seq_length)

        # Process the batch through the model
        model.eval()
        with torch.no_grad():
            outputs = model(batch_tensor)

        # Convert output tensors to text
        output_texts = tensor_to_text(outputs, vocab)

        for input_text, output_text in zip(random_strings, output_texts):
          for i in range(len(input_text)):
            input = input_text[i]
            output = output_text[i]
            if input not in data_out[i]:
              data_out[i][input] = {}
            data_out[i][input][output] = data_out[i][input].get(output, 0) + 1

import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

def plot_advanced_heatmap(data_dict):
    all_characters = sorted(set(data_dict.keys()) | {k for subdict in data_dict.values() for k in subdict.keys()})
    heatmap_data = pd.DataFrame(1, index=all_characters, columns=all_characters)

    for input_char, subdict in data_dict.items():
        for output_char, frequency in subdict.items():
            heatmap_data.at[output_char, input_char] = frequency + 1

    # Applying logarithmic transformation and adjust for log(1) = 0
    log_data = np.log(heatmap_data)

    # Ensure there's a valid range for vmin and vmax
    vmin = log_data.min().min()
    vmax = log_data.max().max()
    if vmin == vmax or vmin <= 0:
        vmin, vmax = 0.1, 1  # Default values to avoid invalid range

    # Defining a custom colormap that changes more in the lower range
    colors = ["blue", "green", "yellow", "red"]  # More intense changes at lower values
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)  # Log normalization

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(log_data, cmap=cmap, norm=norm, annot=True, linewidths=0.5, linecolor='gray')
    ax.set_title('Advanced Log-Scaled Character Transformation Frequencies')
    ax.set_xlabel('Input Characters')
    ax.set_ylabel('Output Characters')

    # Add gridlines after the fact to ensure they are visible on top of the heatmap
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(len(all_characters)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(all_characters)) + 0.5, minor=True)
    ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.show()

# Need to plot for each input. Run cell multiple times with different index 0...31
plot_advanced_heatmap(data_out[0])