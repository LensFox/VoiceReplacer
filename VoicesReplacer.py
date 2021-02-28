import torch
import random
import time

from AudioFileLoader import open_file, save_file
from DataPreparer import prepare_data_loader, get_fragments
from DataProcessor import apply_mask_to_fragments
from NetworkService import NetworkService

audio_path = '../audio/КиШ- Лесник.mp3'
music_path1 = '../audio/Король и Шут - Лесник (минус).mp3'
music_path2 = '../audio/Король и шут - Ели мясо мужики минус.mp3'
music_path3 = '../audio/Король и Шут - Марионетки (минус).mp3'
voice_path1 = '../audio/Король и Шут - Сказка о мертвеце.mp3'
voice_path2 = '../audio/Король и Шут(Страшные сказки) - Солдат и колдун.mp3'
voice_path3 = '../audio/Король и Шут - Сказка про дракона.mp3'

need_train = False
need_create_audio = True

def prepare_data_to_train(file_path, is_voice):
    clear_amplitudes = open_file(file_path)
    data_loader = prepare_data_loader(clear_amplitudes, is_voice)

    return data_loader

network_service = NetworkService(need_train)

if need_train:
    voice_data_loader1 = prepare_data_to_train(voice_path1, True)
    voice_data_loader2 = prepare_data_to_train(voice_path2, True)
    voice_data_loader3 = prepare_data_to_train(voice_path3, True)
    music_data_loader1 = prepare_data_to_train(music_path1, False)
    music_data_loader2 = prepare_data_to_train(music_path2, False)
    music_data_loader3 = prepare_data_to_train(music_path3, False)

    data_loader = voice_data_loader1 + voice_data_loader2 + voice_data_loader3 + music_data_loader1 + music_data_loader2 + music_data_loader3
    random.shuffle(data_loader)

    start_time = time.time()
    network_service.train_network(data_loader)
    end_time = time.time()
    print('trained for {} sec'.format((end_time - start_time) / 1000))

if need_create_audio:
    clear_amplitudes = open_file(audio_path)

    fragments = get_fragments(clear_amplitudes)

    masks_for_fragments = network_service.get_fragments_masks(fragments)

    amplitudes = apply_mask_to_fragments(fragments, masks_for_fragments)

    save_file('voice.wav', amplitudes)

    import matplotlib.pyplot as plt
    import librosa.display
    import numpy as np
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    librosa.display.waveplot(np.array(clear_amplitudes), sr=20050, ax=ax[0])
    ax[0].set(title='Vocal + Instrumental')
    ax[0].label_outer()
    librosa.display.waveplot(np.array(amplitudes), sr=20050, ax=ax[1])
    ax[1].set(title='Vocal')
    ax[1].label_outer()
    plt.show()

#import matplotlib.pyplot as plt
#f = open("loss.txt", "r")
#losses = f.read()
#losses = losses.split(';')
#losses = list(map(float, losses))

#plt.plot(losses)
#plt.show()