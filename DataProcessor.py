import torch

VALUE_THRESHOLD = 0.8
INTERESTED_FRAIM_INDEX = 12

def apply_mask_to_fragments(fragments, masks):
    amplitudes = []

    max_element = max([
        max(list(map(lambda tens: torch.max(tens).tolist(), masks))), 
        abs(min(list(map(lambda tens: torch.min(tens).tolist(), masks))))])
    for i in range(len(fragments)):
        fraim = fragments[i][0][0][INTERESTED_FRAIM_INDEX].tolist()
        mask = masks[i].tolist()
        mask = [1 if x >= VALUE_THRESHOLD else 0 for x in mask]

        result = [fraim[j] * mask[j] for j in range(len(fraim))]

        amplitudes.extend(result)

    return amplitudes
