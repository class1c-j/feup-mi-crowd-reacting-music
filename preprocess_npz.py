import csv
import pandas as pd
import numpy as np
import os
from pypianoroll import Multitrack, Track, load

labeled_csv_path = "./data/labeled_lpd.csv"

df = pd.read_csv(
    labeled_csv_path, skiprows=1, header=None, names=["Path", "Valence", "Energy"]
)


def calculate_quadrant(valence, energy):
    return (
        1
        if valence < 0.5 and energy >= 0.5
        else 2
        if valence >= 0.5 and energy >= 0.5
        else 3
        if valence < 0.5 and energy < 0.5
        else 4
    )


df["Quadrant"] = df.apply(
    lambda row: calculate_quadrant(row["Valence"], row["Energy"]), axis=1
)
df = df.drop(["Valence", "Energy"], axis=1)

quadrant_1_paths = df[df["Quadrant"] == 1]["Path"].tolist()
quadrant_2_paths = df[df["Quadrant"] == 2]["Path"].tolist()
quadrant_3_paths = df[df["Quadrant"] == 3]["Path"].tolist()
quadrant_4_paths = df[df["Quadrant"] == 4]["Path"].tolist()


import numpy as np
import scipy.sparse as sp


def load_npz(file_path):
    data = np.load(file_path)
    pianoroll_0 = sp.csc_matrix(
        (
            data["pianoroll_0_csc_data"],
            data["pianoroll_0_csc_indices"],
            data["pianoroll_0_csc_indptr"],
        ),
        shape=data["pianoroll_0_csc_shape"],
    )
    pianoroll_1 = sp.csc_matrix(
        (
            data["pianoroll_1_csc_data"],
            data["pianoroll_1_csc_indices"],
            data["pianoroll_1_csc_indptr"],
        ),
        shape=data["pianoroll_1_csc_shape"],
    )
    pianoroll_2 = sp.csc_matrix(
        (
            data["pianoroll_2_csc_data"],
            data["pianoroll_2_csc_indices"],
            data["pianoroll_2_csc_indptr"],
        ),
        shape=data["pianoroll_2_csc_shape"],
    )
    pianoroll_3 = sp.csc_matrix(
        (
            data["pianoroll_3_csc_data"],
            data["pianoroll_3_csc_indices"],
            data["pianoroll_3_csc_indptr"],
        ),
        shape=data["pianoroll_3_csc_shape"],
    )
    pianoroll_4 = sp.csc_matrix(
        (
            data["pianoroll_4_csc_data"],
            data["pianoroll_4_csc_indices"],
            data["pianoroll_4_csc_indptr"],
        ),
        shape=data["pianoroll_4_csc_shape"],
    )
    pianoroll_5 = sp.csc_matrix(
        (
            data["pianoroll_5_csc_data"],
            data["pianoroll_5_csc_indices"],
            data["pianoroll_5_csc_indptr"],
        ),
        shape=data["pianoroll_5_csc_shape"],
    )
    pianoroll_6 = sp.csc_matrix(
        (
            data["pianoroll_6_csc_data"],
            data["pianoroll_6_csc_indices"],
            data["pianoroll_6_csc_indptr"],
        ),
        shape=data["pianoroll_6_csc_shape"],
    )
    pianoroll_7 = sp.csc_matrix(
        (
            data["pianoroll_7_csc_data"],
            data["pianoroll_7_csc_indices"],
            data["pianoroll_7_csc_indptr"],
        ),
        shape=data["pianoroll_7_csc_shape"],
    )
    pianoroll_8 = sp.csc_matrix(
        (
            data["pianoroll_8_csc_data"],
            data["pianoroll_8_csc_indices"],
            data["pianoroll_8_csc_indptr"],
        ),
        shape=data["pianoroll_8_csc_shape"],
    )
    pianoroll_9 = sp.csc_matrix(
        (
            data["pianoroll_9_csc_data"],
            data["pianoroll_9_csc_indices"],
            data["pianoroll_9_csc_indptr"],
        ),
        shape=data["pianoroll_9_csc_shape"],
    )
    pianoroll_10 = sp.csc_matrix(
        (
            data["pianoroll_10_csc_data"],
            data["pianoroll_10_csc_indices"],
            data["pianoroll_10_csc_indptr"],
        ),
        shape=data["pianoroll_10_csc_shape"],
    )
    pianoroll_11 = sp.csc_matrix(
        (
            data["pianoroll_11_csc_data"],
            data["pianoroll_11_csc_indices"],
            data["pianoroll_11_csc_indptr"],
        ),
        shape=data["pianoroll_11_csc_shape"],
    )

    return {
        "pianoroll_0": pianoroll_0,
        "pianoroll_1": pianoroll_1,
        "pianoroll_2": pianoroll_2,
        "pianoroll_3": pianoroll_3,
        "pianoroll_4": pianoroll_4,
        "pianoroll_5": pianoroll_5,
        "pianoroll_6": pianoroll_6,
        "pianoroll_7": pianoroll_7,
        "pianoroll_8": pianoroll_8,
        "pianoroll_9": pianoroll_9,
        "pianoroll_10": pianoroll_10,
        "pianoroll_11": pianoroll_11,
    }


data = load_npz(quadrant_1_paths[0])
import pypianoroll

multitracks = []
for key in data:
    pianoroll = data[key].toarray()
    track = pypianoroll.Track(pianoroll=pianoroll)
    multitrack = pypianoroll.Multitrack(tracks=[track])
    multitracks.append(multitrack)

print(multitracks)
