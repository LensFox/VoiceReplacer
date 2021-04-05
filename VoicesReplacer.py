import torch
import random

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

FILES_PER_BLOCK = 3
EPOCHES_COUNT = 20
EPOCHES_PER_BLOCK = 5
GLOBAL_EPOCHES_COUNT = EPOCHES_COUNT // EPOCHES_PER_BLOCK

network_service = NetworkService(need_train)
dataset_preparer_service = DatasetPreparerService()

if need_train:
    feature_file_names = get_all_file_names_in_folder(features_folder_path)
    mask_file_names = get_all_file_names_in_folder(masks_folder_path)
    assert len(feature_file_names) == len(mask_file_names)

    file_names = [(feature_file_names[i], mask_file_names[i]) for i in range(len(feature_file_names))]
    
    file_block_count = len(file_names) // FILES_PER_BLOCK + 1

    # train N-times in different files with downloading features
    for global_epoch_index in range(GLOBAL_EPOCHES_COUNT):
        random.shuffle(file_names)

        # train each block
        for block_index in range(file_block_count):
            file_block = file_names[block_index * FILES_PER_BLOCK: (block_index + 1) * FILES_PER_BLOCK]

            data_loader = dataset_preparer_service.prepare_data_to_train(file_block)

            # train each block M-times to prevent many readings from disk
            for epoch_number in range(EPOCHES_PER_BLOCK):
                random.shuffle(data_loader)

                network_service.train_network(
                    data_loader, 
                    global_epoch_index, 
                    GLOBAL_EPOCHES_COUNT,
                    epoch_number,
                    EPOCHES_PER_BLOCK,
                    block_index,
                    file_block_count)


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