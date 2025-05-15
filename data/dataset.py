import requests
import json
import yt_dlp
import csv
import itertools

def main():
    with open('id2.csv') as f:
        ids = list(itertools.chain.from_iterable(csv.reader(f)))

    get_chord_progression(ids)
    get_audio(ids)

def get_chord_progression(ids):
    for id in ids:
        chord_url = f"https://widget.songle.jp/api/v1/song/chord.json?url=www.youtube.com/watch?v={id}"
        res = requests.get(chord_url)
        data = json.loads(res.text)

        with open(f"chord/{id}.json", mode="w") as f:
            json.dump(data, f, indent=2)

def get_audio(ids):
    for id in ids:
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': f'audio/{id}',
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                'preferredquality': '128',
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([f"https://www.youtube.com/watch?v={id}"])
            except:
                None

if __name__ == '__main__':
    main()