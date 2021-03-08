import torch

FRAIM_SIZE = 513
FRAGMENT_SIZE = 25
VALUE_THRESHOLD = 1e-4

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
        new_fragment = [[new_fragment]]
        fragments.append(new_fragment)

    return fragments


def get_fragments(clear_amplitudes):
    amplitudes = crop_array_to_fit_block_size(clear_amplitudes)

    fraims = make_fraims(amplitudes)

    fragments = zip_fraims_to_fragments(fraims)
    fragments = torch.Tensor(fragments)

    return fragments

def get_masks(mask_amplitudes):
    amplitudes = crop_array_to_fit_block_size(mask_amplitudes)
    
    fraims = make_fraims(amplitudes)
    fraims = fraims[12 : -13]
    for i in range(len(fraims)):
        for j in range(FRAIM_SIZE):
            fraims[i][j] = 1 if abs(fraims[i][j]) >= VALUE_THRESHOLD else 0
    fraims = torch.Tensor(fraims).reshape((-1, 1, FRAIM_SIZE))

    return fraims


def prepare_data_loader(clear_amplitudes, mask_amplitudes):
    feature_fragments = get_fragments(clear_amplitudes)
    mask_fraims = get_masks(mask_amplitudes)

    data_loader = [(feature_fragments[i], mask_fraims[i]) for i in range(len(mask_fraims))]

    return data_loader
