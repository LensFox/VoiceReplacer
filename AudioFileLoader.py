import librosa
import soundfile as sf
from time import time
import os

SAMPLE_RATE = 22050

def open_file(file_path):
    print('loading {}'.format(file_path))
    start_time = time()
    amplitudes, _ = librosa.load(file_path)
    end_time = time()

    print('The file {} has been loaded for {} s'.format(file_path, end_time - start_time))

    return amplitudes.tolist()

def save_file(file_path, amplitudes):
    sf.write(file_path, amplitudes, SAMPLE_RATE)

    print('The file {} has been saved'.format(file_path))

def get_all_file_names_in_folder(folder_path):
    file_names = os.listdir(folder_path)
    file_names = list(map(lambda file_name: os.path.join(folder_path, file_name), file_names))

    return file_names