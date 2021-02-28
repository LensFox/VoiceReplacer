import torch

FRAIM_SIZE = 513
FRAGMENT_SIZE = 25

def filter_empty_amplitudes(amplitudes, precision = 1e-4):
    return list(filter(lambda x: 1 if abs(x) > precision else 0, amplitudes))

def crop_array_to_fit_block_size(amplitudes, aliquot_size):
    new_size = len(amplitudes) // aliquot_size * aliquot_size
    return amplitudes[:new_size]

def make_fraims(amplitudes, block_size):
    return torch.Tensor(amplitudes).view(-1, block_size).tolist()

def zip_fraims_to_fragments(fraims, fragment_size):
    fragments = []

    fraims_count = len(fraims)

    for i in range(0, fraims_count - fragment_size):
        new_fragment = fraims[i : i + fragment_size]
        new_fragment = [[new_fragment]]
        fragments.append(new_fragment)

    return fragments

def make_data_loader(fragments, mask):
    data_loader = [(fragments[i], mask[i]) for i in range(len(mask))]

    return data_loader

def get_fragments(clear_amplitudes):
    amplitudes = filter_empty_amplitudes(clear_amplitudes)
    amplitudes = crop_array_to_fit_block_size(amplitudes, FRAIM_SIZE)

    fraims = make_fraims(amplitudes, FRAIM_SIZE)

    fragments = zip_fraims_to_fragments(fraims, FRAGMENT_SIZE)
    fragments = torch.Tensor(fragments)

    return fragments

def prepare_data_loader(clear_amplitudes, is_voice):
    fragments = get_fragments(clear_amplitudes)
    mask_for_voice = torch.Tensor([1 if is_voice else 0] * fragments.size(0))
    mask_for_voice = torch.reshape(mask_for_voice, (-1, 1, 1))

    data_loader = make_data_loader(fragments, mask_for_voice)

    return data_loader