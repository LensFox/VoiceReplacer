import librosa
import soundfile as sf

SAMPLE_RATE = 22050

def open_file(file_path):
    amplitudes, _ = librosa.load(file_path)
    
    print('The file {} has been loaded'.format(file_path))

    return amplitudes.tolist()

def save_file(file_path, amplitudes):
    sf.write(file_path, amplitudes, SAMPLE_RATE)

    print('The file {} has been saved'.format(file_path))
