from gtts import gTTS
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def text_to_audio(text, topic="ml_topic"):
    """
    Converts a string of text into an MP3 audio file using gTTS.

    Args:
        text: Text to convert to audio
        topic: Topic name for filename

    Returns:
        Filename of generated audio file or None if failed
    """

    if not text or not text.strip():
        logger.warning("Cannot generate audio from empty text.")
        return None

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:30]
        filename = f"{safe_topic}_{timestamp}.mp3"
        filepath = os.path.join("generated_audio", filename)

        # Generate audio
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)

        logger.info(f"Audio generated successfully: {filename}")
        return filename

    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        return None


def delete_old_audio_files(max_age_hours=24):
    """
    Delete audio files older than specified hours

    Args:
        max_age_hours: Maximum age of files to keep in hours
    """

    try:
        audio_dir = "generated_audio"
        if not os.path.exists(audio_dir):
            return

        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600

        for filename in os.listdir(audio_dir):
            filepath = os.path.join(audio_dir, filename)

            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)

                if file_age > max_age_seconds:
                    os.remove(filepath)
                    logger.info(f"Deleted old audio file: {filename}")

    except Exception as e:
        logger.error(f"Error deleting old audio files: {e}")