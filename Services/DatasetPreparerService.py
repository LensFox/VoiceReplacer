from time import time
import torch

from AudioFileLoader import open_file, save_file
from DataPreparer import get_feature_fragments, get_mask_fraims, prepare_data_loader, FRAIM_SIZE, FRAGMENT_SIZE

class DatasetPreparerService:

    def prepare_data_to_train(self, file_names):
        feature_fragments = []
        mask_fraims = []

        for feature_path, mask_path in file_names:
            feature_amplitudes = open_file(feature_path)
            mask_amplitudes = open_file(mask_path)

            current_feature_fragments = get_feature_fragments(feature_amplitudes)
            current_mask_fraims = get_mask_fraims(mask_amplitudes)

            feature_fragments.extend(current_feature_fragments)
            mask_fraims.extend(current_mask_fraims)

        print('Making data for dataset started')
        time_before_batching = time()
        data_loader = prepare_data_loader(feature_fragments, mask_fraims)
        print('Making data for dataset ended for {}'.format(time() - time_before_batching))

        return data_loader

    def prepare_audio_to_predict(self, audio_path):
        feature_amplitudes = open_file(audio_path)

        fragments = get_feature_fragments(feature_amplitudes)
        fragments = torch.Tensor(fragments).view(-1, 1, 1, FRAGMENT_SIZE, FRAIM_SIZE)

        return (feature_amplitudes, fragments)
