import os
from time import time
import torch

from AudioFileLoader import open_file, save_file
from DataPreparer import get_feature_fragments, get_mask_fraims, prepare_data_loader, FRAIM_SIZE, FRAGMENT_SIZE

class DatasetPreparerService:

    def prepare_data_to_train(self, features_folder, masks_folder):
        feature_file_names = self.__get_all_file_names_in_folder(features_folder)
        mask_file_names = self.__get_all_file_names_in_folder(masks_folder)
        assert len(feature_file_names) == len(mask_file_names)

        feature_fragments = []
        mask_fraims = []

        for i in range(len(feature_file_names)):
            feature_path = feature_file_names[i]
            mask_path = mask_file_names[i]

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


    def __get_all_file_names_in_folder(self, folder_path):
        file_names = os.listdir(folder_path)
        file_names = list(map(lambda file_name: os.path.join(folder_path, file_name), file_names))

        return file_names