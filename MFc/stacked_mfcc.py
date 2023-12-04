import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Folder containing audio files
audio_folder = "E:\\ML project\\codes\\data\\LJSpeech-1.1\\wavs"

# Set the desired number of frames
desired_frames = 500

# Initialize an empty list to store the results
mfcc_list = []
i = 0

for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):  # assuming all audio files are in WAV format
        audio_path = os.path.join(audio_folder, filename)

        # Load the audio signal
        signal, sample_rate = librosa.load(audio_path, sr=None)
        if i % 100 == 0:
            print("iteration ", i)
        i += 1
        # Compute MFCCs with zero-padding
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,  # Sampling Rate
            n_mfcc=40,  # Number of MFCCs
            n_fft=2048,  # Number of FFT Points
            hop_length=512,  # Hop Length
            pad_mode="constant",  # Padding Mode
            n_mels=128,  # Number of Mel Filterbanks
        )

        # If the number of frames is less than desired_frames, zero-pad
        if mfcc.shape[1] < desired_frames:
            pad_width = desired_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))

        # If the number of frames is greater than desired_frames, truncate
        elif mfcc.shape[1] > desired_frames:
            mfcc = mfcc[:, :desired_frames]

        # Append the current MFCCs to the list
        mfcc_list.append(mfcc)

# Stack all the MFCCs along a new axis (axis 0)
stacked_mfccs = np.stack(mfcc_list, axis=0)

# # Visualize one of the results (for demonstration purposes)
# plt.figure(figsize=(10, 5))
# librosa.display.specshow(stacked_mfccs[0], x_axis="time")
# plt.colorbar()
# plt.show()

# Display the shape of the stacked MFCCs
print("Shape of stacked MFCCs:", stacked_mfccs.shape)
np.save("data_40_500", stacked_mfccs)
