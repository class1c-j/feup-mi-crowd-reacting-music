import os
import csv
from tqdm import tqdm

# Path to the dataset folder
dataset_folder = "./data/lpd_5/lpd_5_cleansed/"

# Path to the cleansed_ids.txt file
cleansed_ids_file = "./data/lpd/cleansed_ids.txt"

# Path to the unique_tracks.txt file
unique_tracks_file = "./data/lpd/unique_tracks.txt"

# Path to save the output CSV file
output_csv_file = "./data/named_lpd.csv"

# Read the cleansed_ids.txt file into a dictionary
id_mapping = {}
with open(cleansed_ids_file, "r", encoding="UTF-8") as f:
    for line in f:
        parts = [i for i in line.strip().split(" ") if i != ""]
        if len(parts) == 2:
            id_mapping[parts[0]] = parts[1]


# Read the unique_tracks.txt file into a dictionary
track_mapping = {}
with open(unique_tracks_file, "r", encoding="UTF-8") as f:
    for line in f:
        parts = line.strip().split("<SEP>")
        if len(parts) == 4:
            track_mapping[parts[0]] = (parts[2], parts[3])


total_files = sum(
    1
    for _, _, files in os.walk(dataset_folder)
    for file in files
    if file.endswith(".npz")
)


# Iterate through the dataset folder
output_data = []
with tqdm(total=total_files, desc="Processing") as pbar:
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            if filename.endswith(".npz"):
                npz_id = os.path.splitext(filename)[0]
                npz_path = os.path.join(root, filename)
                if npz_id in id_mapping and id_mapping[npz_id] in track_mapping:
                    artist, track_title = track_mapping[id_mapping[npz_id]]
                    output_data.append([npz_path, artist, track_title])
                pbar.update(1)

# Write the output data to the CSV file
with open(output_csv_file, "w", newline="", encoding="UTF-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Path to NPZ", "Artist", "Track Title"])
    writer.writerows(output_data)

print("CSV file generated successfully.")
