# youtube_utils.py

import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

def get_video_id(url: str) -> str | None:
    """Extracts the YouTube video ID from a URL using regex."""
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id: str) -> list | None:
    """Fetches the English transcript for a given YouTube video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])
        return transcript.fetch()
    except NoTranscriptFound:
        print(f"No English transcript found for video ID {video_id}.")
        return None
    except Exception as e:
        print(f"An error occurred while fetching the transcript: {e}")
        return None

def format_transcript_as_text(transcript: list) -> str:
    """Formats the raw transcript data into a single, clean string."""
    if not transcript:
        return ""
    return " ".join([item['text'] for item in transcript])