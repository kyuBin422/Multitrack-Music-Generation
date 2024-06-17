from googleapiclient.discovery import build
import json
from youtubesearchpython import VideosSearch
import pickle
from functools import reduce

api_key = 'AIzaSyCdwP06YLzhA38GCT3kTnur4XypVAaJVwU'


def video_comments(video_id):
    # empty list for storing reply
    replies = []

    # creating youtube resource object
    youtube = build('youtube', 'v3',
                    developerKey=api_key)

    # retrieve youtube video results
    video_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id
    ).execute()

    # extracting required info
    # from each result object
    for item in video_response['items']:

        # Extracting comments
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        # counting number of reply of comment
        replycount = item['snippet']['totalReplyCount']

        # if reply is there
        if replycount > 0:

            # iterate through all reply
            for reply in item['replies']['comments']:
                # Extract reply
                reply = reply['snippet']['textDisplay']

                # Store reply is list
                replies.append(reply)
        else:
            replies.append(comment)

        # # print comment with list of reply
        # print(comment, replies, end='\n\n')
    return replies


def video_url(key_val):
    videosSearch = VideosSearch(key_val, limit=1)
    return videosSearch.result()['result'][0]['id']


# Call function
# video_comments(video_id)

def getComment():
    with open('data/md5_to_paths.json') as f:
        md5_to_paths = json.load(f)

    NameList = set()
    for key in list(md5_to_paths.values()):
        for subkey in key:
            NameList.add(subkey.split('.')[0])

    with open('data/42000.pickle', 'rb') as f:
        Name2Comment = pickle.load(f)
    for idx, name in enumerate(NameList):
        print(idx)
        # Enter video id
        if idx > 42000:
            try:
                video_id = video_url(name)
                replies = video_comments(video_id)
            except Exception as e:
                print(e)
                replies = []
            Name2Comment[name] = replies
            if idx % 1000 == 0:
                with open('data/' + str(idx) + '.pickle', 'wb') as handle:
                    pickle.dump(Name2Comment, handle, protocol=pickle.HIGHEST_PROTOCOL)
