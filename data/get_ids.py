import csv
import requests
import os

API_KEY = os.environ["YOUTUBE_API_KEY"]

def get_ids_from_playlist(playlist_id, page_token=None):
    params = {
        'part': 'snippet',
        'playlistId': playlist_id,
        'maxResults': 50,
        'key': API_KEY,
        'pageToken': page_token
    }
    response = requests.get('https://www.googleapis.com/youtube/v3/playlistItems', params=params)
    data = response.json()
    return data.get('items', []), data.get('nextPageToken')

def get_all_ids_from_playlist(playlist_id):
    all_video_ids = []
    next_page_token = None

    while True:
        items, next_page_token = get_ids_from_playlist(playlist_id, next_page_token)
        video_ids = [item['snippet']['resourceId']['videoId'] for item in items]
        all_video_ids.extend(video_ids)

        if not next_page_token:
            break

    return all_video_ids

def write_to_id_csv():
    playlist_ids = ["PL6c6sPNdnX_UjsnvrQ_fssRHcon05f0Xd", "PLSf-HCzj7cOv511WfVw6ijNDgUu8eQZTx", "PLSXruysuXxNgvP3I_1-6ACwpjkS3HakJx", 
                    "PLtJnHhA9MVicF2uRb7zfOqX34cBpCCdXE", "PLRK6d_2i1MytdbaZYma1zRYWc4YC0IFTN", "PLxKNZPFp--_CjWWCR3Abt79-Y5u6FPmd2"]

    with open('id.csv', 'w') as f:
        writer = csv.writer(f)
        for playlist_id in playlist_ids:
            ids = get_all_ids_from_playlist(playlist_id)
            for id in ids:
                writer.writerow([id])

if __name__ == '__main__':
    write_to_id_csv()

        