import random
import os
import torch

from AudioFileLoader import open_file, save_file
from DataPreparer import get_fragments, get_masks, FRAGMENT_SIZE, FRAIM_SIZE

class DatasetPreparerService:

    def prepare_data_to_train(self, features_folder, masks_folder, batch_size = 100):
        feature_file_names = self.__get_all_file_names_in_folder(features_folder)
        mask_file_names = self.__get_all_file_names_in_folder(masks_folder)
        assert len(feature_file_names) == len(mask_file_names)

        fragments_loader = []
        masks_loader = []

        for i in range(len(feature_file_names)):
            feature_path = feature_file_names[i]
            mask_path = mask_file_names[i]

            feature_amplitudes = open_file(feature_path)
            mask_amplitudes = open_file(mask_path)

            current_fragments_loader = get_fragments(feature_amplitudes)
            current_masks_loader = get_masks(mask_amplitudes)

            fragments_loader = fragments_loader + current_fragments_loader
            masks_loader = masks_loader + current_masks_loader
            
        remainder_size = len(fragments_loader) % batch_size
        fragments_loader = fragments_loader[:-remainder_size]
        masks_loader = masks_loader[:-remainder_size]
        fragments_loader = torch.Tensor(fragments_loader).reshape((-1, batch_size, 1, FRAGMENT_SIZE, FRAIM_SIZE)).tolist()
        masks_loader = torch.Tensor(masks_loader).reshape((-1, batch_size, FRAIM_SIZE)).tolist()

        data_loader = [(torch.Tensor(fragments_loader[i]), torch.Tensor(masks_loader[i])) for i in range(len(fragments_loader))]
        random.shuffle(data_loader)

        return data_loader

    def prepare_audio_to_predict(self, audio_path):
        feature_amplitudes = open_file(audio_path)

        fragments = get_fragments(feature_amplitudes)

        return (feature_amplitudes, fragments)


    def __get_all_file_names_in_folder(self, folder_path):
        file_names = os.listdir(folder_path)
        file_names = list(map(lambda file_name: os.path.join(folder_path, file_name), file_names))

        return file_names