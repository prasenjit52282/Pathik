import datetime
import pandas as pd
from librosa import load
import librosa.display
from librosa.feature import mfcc
from scipy.io import wavfile as wav
from .constants import sr
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from collections import Counter
import os

traffic_frequency_range = [500,2500]

def get_date_time_str(start_time):
    datetime_string=\
    '{:02d}/{:02d}/{:04d} {:02d}:{:02d}:{:02d}'\
    .format(start_time.month,
            start_time.day,
            start_time.year,
            start_time.hour,
            start_time.minute,
            start_time.second)
    return datetime_string


def process_audio_file(filename):
    #sr = 8000 #for our case sampling rate is 8000 Hz @Imported from constants
    x, _ = load(filename,sr=sr) # using librosa
    
    _,_,year,month,day,hour,minute,second,_=filename.split("/")[-1].split('_')
    start_time=datetime.datetime(int(year),
                                 int(month),
                                 int(day),
                                 int(hour), 
                                 int(minute),
                                 int(second))
    
    x_slice=[]
    sec_wise_data=[]
    x_len=(x.shape[0]//sr)*sr #full samples

    for i in range(x_len):
        x_slice.append(x[i])
        if (i+1)%sr==0:
            sec_wise_data.append([get_date_time_str(start_time)]+x_slice)
            x_slice=[]
            start_time+=datetime.timedelta(seconds=1) #adding one full second
    
    df=pd.DataFrame(sec_wise_data,columns=['time']+[f'amp_{i}' for i in range(0,sr)])
        
    return df #returns audio second wise and have 8000(sr) columns

def MFCC_on_Array(array):
    #sr = 8000 #for our case sampling rate is 8000 Hz  @Imported from constants
    MFCC_COMPONENT=5 #top 5 mfcc components are taken
    mfccs = mfcc(y = array, sr=sr) #using librosa
    rms = np.sqrt(np.mean(np.square(array)))
     # Compute the intensity of the signal
    intensity = rms**2 / (2 * np.pi**2)
    # Compute the sound pressure level (SPL) in decibels (dB)
    spl = 20 * np.log10(intensity / (2e-5))
    return sorted(list(mfccs.mean(axis=1)),reverse=True)[:MFCC_COMPONENT] 

def get_intensity_from_array(array):
    S, _ = librosa.magphase(librosa.stft(array,n_fft = sr))
    avg_power = np.square(S[2300:,:]).mean()
    loudness_in_db = 10*np.log(avg_power)
    return loudness_in_db 

def plotFrequency(sampling_freq, audio):
    # Calculate the Fast Fourier Transform (FFT) of the audio signal
    fft = np.fft.fft(audio)

    # Calculate the magnitude of the FFT
    magnitude = np.abs(fft)

    # Create an array of frequency values
    freqs = np.fft.fftfreq(len(fft), 1/sampling_freq)

    # Find the indices of the positive frequencies
    positive_freqs = freqs[:len(freqs)//2]

    # Plot the frequency vs frequency count
    plt.plot(positive_freqs, magnitude[:len(freqs)//2])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency count")
    plt.show()

def extract_traffic_noise(file):
    # Load the WAV file
    # sample_rate, audio_data = wav.read(file)
    audio_data, sample_rate = librosa.load(file,sr=sr)

    # mfcc on sound freq 
    top5freq, _ = MFCC_on_Array(audio_data)
    print("mfcc before filtering: ",top5freq)
    # plotFrequency(sample_rate, audio_data)

    # Convert the audio data to mono if it is stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Set the window size for spectral analysis
    window_size = 2048

    # Compute the power spectrum of the audio data
    frequencies, times, power_spectrum = signal.stft(audio_data, fs=sample_rate, window='hann', nperseg=window_size)

    # Identify the frequency range of the traffic noise
    min_freq = 2000
    max_freq = 3300
    noise_indices = np.where((frequencies <= min_freq) | (frequencies >= max_freq))[0]

    # Create a filter to remove the traffic noise
    noise_filter = np.ones_like(power_spectrum)
    noise_filter[noise_indices] = 0

    # Apply the filter to the power spectrum
    filtered_spectrum = power_spectrum * noise_filter

    # Reconstruct the audio data from the filtered spectrum
    filtered_audio_data = signal.istft(filtered_spectrum, fs=sample_rate, window='hann', nperseg=window_size)[1]
    # plotFrequency(sample_rate, filtered_audio_data)

    # mfcc on filtered sound freq 
    print(len(filtered_audio_data))
    top5freq, _ = MFCC_on_Array(filtered_audio_data)
    print("mfcc after filtering: ",top5freq)

    mfcc_components_for_each_second(filtered_audio_data, sample_rate)

    parent_directory = os.path.dirname(os.getcwd())
    output_file = parent_directory + "/Output/traffic_noise_file.wav"
    # Save the filtered audio data to a new WAV file
    wav.write(output_file, sample_rate, filtered_audio_data.astype(np.int16))

def mfcc_components_for_each_second(audio_data, sampling_rate):
    context_data = np.array([])
    context_time = 60   # 60 seconds audio context is considered
    total_seconds = len(audio_data)//sampling_rate
    time =0
    print(audio_data[time*sampling_rate: (time+1)*sampling_rate])
    for time in range(0, total_seconds):
        context_data =np.append(context_data, audio_data[time*sampling_rate: (time+1)*sampling_rate])
        while(len(context_data) > context_time*sampling_rate):
            context_data = context_data[sampling_rate: ]
        
        print(f"MFCC components for time {time}",MFCC_on_Array(context_data), len(context_data))

# def intensity_time_plot(file):
#     y, sample_rate = librosa.load(file,sr=sr)
#     S, phase = librosa.magphase(librosa.stft(y))
#     rms = librosa.feature.rms(S=S)
#     fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
#     times = librosa.times_like(rms)
#     ax[0].semilogy(times, rms[0], label='RMS Energy')
#     ax[0].set(xticks=[])
#     ax[0].legend()
#     ax[0].label_outer()
#     librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
#                             y_axis='log', x_axis='time', ax=ax[1])
#     ax[1].set(title='log Power spectrogram')
#     plt.show()
