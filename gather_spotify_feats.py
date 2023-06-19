import os
import csv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import time

client_id = "<CLIENT_ID>"
client_secret = "<CLIENT_SECRET>"

input_csv_file = "./data/named_lpd.csv"

output_csv_file = "./data/labeled_lpd.csv"

client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

track_data = []
with open(input_csv_file, "r", encoding="UTF-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        npz_path, artist, track_title = row
        track_data.append((npz_path, artist, track_title))


with open(output_csv_file, "w", newline="", encoding="UTF-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Path", "Valence", "Energy"])

    with tqdm(total=len(track_data), desc="Processing") as pbar:
        for npz_path, artist, track_title in track_data:
            query = f"artist:{artist} track:{track_title}"
            results = sp.search(q=query, type="track", limit=1)
            if (
                results
                and "tracks" in results
                and "items" in results["tracks"]
                and len(results["tracks"]["items"]) > 0
            ):
                track = results["tracks"]["items"][0]
                track_id = track["id"]
                audio_features = sp.audio_features(track_id)
                if audio_features and audio_features[0] is not None:
                    valence = audio_features[0]["valence"]
                    energy = audio_features[0]["energy"]
                    writer.writerow([npz_path, valence, energy])
            pbar.update(1)
            time.sleep(0.1)

print("CSV file generated successfully.")
