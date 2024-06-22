import torch
import torch.nn.functional as F

def one_hot_encode(column):
    string_to_index = {string: index for index, string in enumerate(set(column))}

    # Convert the list to a list of integers using the dictionary
    index_list = [string_to_index[string] for string in column]

    # Convert the list of integers to a one-hot encoded tensor
    one_hot_encoded = F.one_hot(torch.tensor(index_list))
    return one_hot_encoded