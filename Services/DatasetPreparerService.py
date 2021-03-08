import os

from AudioFileLoader import open_file, save_file
from DataPreparer import prepare_data_loader, get_fragments

class DatasetPreparerService:

    def prepare_data_to_train(self, features_folder, masks_folder):
        feature_file_names = self.__get_all_file_names_in_folder(features_folder)
        mask_file_names = self.__get_all_file_names_in_folder(masks_folder)
        assert len(feature_file_names) == len(mask_file_names)

        data_loader = []

        for i in range(len(feature_file_names)):
            feature_path = feature_file_names[i]
            mask_path = mask_file_names[i]

            feature_amplitudes = open_file(feature_path)
            mask_amplitudes = open_file(mask_path)

            current_data_loader = prepare_data_loader(feature_amplitudes, mask_amplitudes)
            data_loader = data_loader + current_data_loader

        return data_loader

    def prepare_audio_to_predict(self, audio_path):
        feature_amplitudes = open_file(audio_path)

        fragments = get_fragments(feature_amplitudes)

        return (feature_amplitudes, fragments)


    def __get_all_file_names_in_folder(self, folder_path):
        file_names = os.listdir(folder_path)
        file_names = list(map(lambda file_name: os.path.join(folder_path, file_name), file_names))

        return file_names