import torch

from AudioFileLoader import save_file
from DataProcessor import apply_mask_to_fragments
from Services.NetworkService import NetworkService
from Services.DatasetPreparerService import DatasetPreparerService
from AudioFileLoader import get_all_file_names_in_folder

audio_path = '../audio/КиШ- Лесник.mp3'
features_folder_path = '../audio/features'
masks_folder_path = '../audio/masks'

need_train = True
need_create_audio = False
need_print_loss = False

network_service = NetworkService(need_train)
dataset_preparer_service = DatasetPreparerService()

if need_train:
    feature_file_names = get_all_file_names_in_folder(features_folder_path)
    mask_file_names = get_all_file_names_in_folder(masks_folder_path)
    assert len(feature_file_names) == len(mask_file_names)

    network_service.train_network(dataset_preparer_service, feature_file_names, mask_file_names)



if need_create_audio:
    feature_amplitudes, fragments = dataset_preparer_service.prepare_audio_to_predict(audio_path)

    masks_for_fragments = network_service.get_fragments_masks(fragments)

    amplitudes = apply_mask_to_fragments(fragments, masks_for_fragments)

    save_file('voice.wav', amplitudes)

    import matplotlib.pyplot as plt
    import librosa.display
    import numpy as np
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    librosa.display.waveplot(np.array(feature_amplitudes), sr=20050, ax=ax[0])
    ax[0].set(title='Vocal + Instrumental')
    ax[0].label_outer()
    librosa.display.waveplot(np.array(amplitudes), sr=20050, ax=ax[1])
    ax[1].set(title='Vocal')
    ax[1].label_outer()
    plt.show()

if need_print_loss:
    import matplotlib.pyplot as plt
    f = open("loss.txt", "r")
    losses = f.read()
    losses = losses.split(';')
    losses = list(map(float, losses))

    plt.plot(losses)
    plt.show()