import torch
import random

FRAIM_SIZE = 513
FRAGMENT_SIZE = 25
VALUE_THRESHOLD = 0.01

def crop_array_to_fit_block_size(amplitudes):
    new_size = len(amplitudes) // FRAIM_SIZE * FRAIM_SIZE
    return amplitudes[:new_size]

def make_fraims(amplitudes):
    return torch.Tensor(amplitudes).view(-1, FRAIM_SIZE).tolist()

def zip_fraims_to_fragments(fraims):
    fragments = []

    fraims_count = len(fraims)

    for i in range(0, fraims_count - FRAGMENT_SIZE):
        new_fragment = fraims[i : i + FRAGMENT_SIZE]
        new_fragment = [new_fragment]
        fragments.append(new_fragment)

    return fragments

def get_feature_fragments(clear_amplitudes):
    amplitudes = crop_array_to_fit_block_size(clear_amplitudes)

    fraims = make_fraims(amplitudes)

    fragments = zip_fraims_to_fragments(fraims)

    return fragments

def get_mask_fraims(mask_amplitudes):
    amplitudes = crop_array_to_fit_block_size(mask_amplitudes)
    
    fraims = make_fraims(amplitudes)
    fraims = fraims[12 : -13]
    for i in range(len(fraims)):
        for j in range(FRAIM_SIZE):
            fraims[i][j] = 1 if abs(fraims[i][j]) >= VALUE_THRESHOLD else 0
    fraims = torch.Tensor(fraims).reshape((-1, FRAIM_SIZE)).tolist()

    return fraims

def prepare_data_loader(feature_fragments, mask_fraims, batch_size = 100):
    feature_loader = []
    mask_loader = []

    unbatched_data_loader = [(feature_fragments[i], mask_fraims[i]) for i in range(len(feature_fragments))]
    random.shuffle(unbatched_data_loader)

    for i in range(len(unbatched_data_loader)):
        if i % batch_size == 0:
            feature_loader.append([])
            mask_loader.append([])

        fragment, mask = unbatched_data_loader[i]

        feature_loader[-1].append(fragment)
        mask_loader[-1].append(mask)

    return [(torch.Tensor(feature_loader[i]), torch.Tensor(mask_loader[i])) for i in range(len(feature_loader))]