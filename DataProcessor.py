import torch

VALUE_THRESHOLD = 0.5
INTERESTED_FRAIM_INDEX = 12

def apply_mask_to_fragments(fragments, masks):
    amplitudes = []

    for i in range(len(fragments)):
        fraim = fragments[i][0][0][INTERESTED_FRAIM_INDEX].tolist()
        mask = masks[i].tolist()
        mask = mask[0]
        mask = [1 if x >= VALUE_THRESHOLD else 0 for x in mask]

        result = [fraim[j] * mask[j] for j in range(len(fraim))]

        amplitudes.extend(result)

    return amplitudes
