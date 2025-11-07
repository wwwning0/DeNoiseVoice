import os

import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import whisper
import jiwer




def compute_MCD(file_original, file_reconstructed):
    def readmgc(filename):
        # all parameters can adjust by yourself :)
        sr, x = wavfile.read(filename)
        # var = sr == 22050
        if x.ndim == 2:
            x = x[:, 0]
        x = x.astype(np.float64)
        frame_length = 1024
        hop_length = 256
        # Windowing
        frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
        frames *= pysptk.blackman(frame_length)
        assert frames.shape[1] == frame_length
        # Order of mel-cepstrum
        order = 25
        alpha = 0.41
        stage = 5
        gamma = -1.0 / stage

        mgc = pysptk.mgcep(frames, order, alpha, gamma)
        mgc = mgc.reshape(-1, order + 1)
        return mgc

    _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    s = 0.0

    framesTot = 0

    # print("Processing original audio----{}".format(file_original))
    mgc1 = readmgc(file_original)
    # print("Processing reconstructed audio----{}".format(file_reconstructed))
    mgc2 = readmgc(file_reconstructed)

    x = mgc1
    y = mgc2

    distance, path = fastdtw(x, y, dist=euclidean)

    distance /= (len(x) + len(y))
    pathx = list(map(lambda l: l[0], path))
    pathy = list(map(lambda l: l[1], path))
    x, y = x[pathx], y[pathy]

    frames = x.shape[0]
    framesTot += frames

    z = x - y
    s += np.sqrt((z * z).sum(-1)).sum()

    MCD_value = _logdb_const * float(s) / float(framesTot)
    return MCD_value


def compute_WER():
    model = whisper.load_model("base")
    # 原始文本
    original_text = "At least 12 persons saw the man with the revolver in the vicinity of the Tipit crime scene, at or immediately after the shooting. By the evening of November 22, five of them had identified Lee Harvey Oswald in police lineups as the man they saw. A sixth did so the next day. Three others subsequently identified Oswald from a photograph. Two witnesses testified that Oswald resembled the man they had seen. One witness felt he was too distant from the gunman to make a positive identification. A taxi driver, William Skoggins, was eating lunch in his cab, which was parked on Patten, facing the southeast corner of Tenth Street and Patten Avenue, a few feet to the north. A police car moving east on 10th at about 10 or 12 miles an hour passed in front of his cab. About 100 feet from the corner, the police car pulled up alongside a man on the sidewalk. This man dressed in a light-colored jacket approached the car."


    file_dir = r"E:\dataset\result\test.wav"
    audio, _ = librosa.load(file_dir, sr=16000)

    result = model.transcribe(audio)
    transcribed_text = result["text"]
    # 计算 WER
    wer_score = jiwer.wer(original_text, transcribed_text)
    print(wer_score)
