#Hate speech detection with each sentence lang det + muting
import subprocess
import whisper
import os
import wave
import requests

PERSPECTIVE_API_KEY = "...."  # Replace with your API key
PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# Extract audio from video
def extract_audio(video_path, audio_path):
    command = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(command, check=True)

# Get audio duration
def get_audio_duration(audio_path):
    with wave.open(audio_path, "rb") as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        return frames / float(rate)

# Perspective API supported languages
ATTRIBUTE_LANG_SUPPORT = {
    "TOXICITY": {"ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "id", "it", "ja", "ko", "pl", "pt", "ru", "es"},
    "SEVERE_TOXICITY": {"de", "en", "es", "fr", "it", "pt", "ru"},
    "IDENTITY_ATTACK": {"de", "en", "es", "fr", "it", "pt", "ru"},
    "INSULT": {"de", "en", "es", "fr", "it", "pt", "ru"},
    "PROFANITY": {"de", "en", "es", "fr", "it", "pt", "ru"},
    "THREAT": {"de", "en", "es", "fr", "it", "pt", "ru"},
}

def get_perspective_toxicity(text, lang):
    """
    Fetches toxicity scores from the Perspective API.

    - If input is a **single word**, only checks 'PROFANITY' (if supported).
    - If input is a **phrase**, checks all supported attributes for the given language.
    - Returns a dictionary of scores or zeros if unsupported.
    """
    text = text.strip()

    # Determine if the input is a single word (no spaces)
    is_single_word = " " not in text

    if is_single_word:
        # If single word, only check 'PROFANITY' if supported
        if lang in ATTRIBUTE_LANG_SUPPORT.get("PROFANITY", []):
            requested_attributes = {"PROFANITY": {}}
        else:
            return {"PROFANITY": 0}  # No API call needed
    else:
        # If phrase, check all supported attributes
        requested_attributes = {attr: {} for attr, langs in ATTRIBUTE_LANG_SUPPORT.items() if lang in langs}

    # If no attributes are valid, return zero scores
    if not requested_attributes:
        return {attr: 0 for attr in ATTRIBUTE_LANG_SUPPORT}

    # Prepare API request
    data = {"comment": {"text": text}, "languages": [lang], "requestedAttributes": requested_attributes}

    

    response = requests.post(f"{PERSPECTIVE_URL}?key={PERSPECTIVE_API_KEY}", json=data)



    # Log the API response
    if response.status_code == 200:
        print(f"🔹 API Response: {response.json()}")
        scores = response.json().get("attributeScores", {})
        return {attr: scores[attr]["summaryScore"]["value"] for attr in scores}
    else:
        print(f"⚠️ API Error: {response.status_code} - {response.text}")
        return {attr: 0 for attr in ATTRIBUTE_LANG_SUPPORT}

# Transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, word_timestamps=True)
    detected_lang = result.get("language", "en")
    return result["segments"], model, detected_lang


import langid








# Analyze hate speech with profanity - with per-sentence language detection
def analyze_hate_speech(transcriptions):
    hate_speech_sentences = []
    word_intervals = {}

    for segment in transcriptions:
        sentence = segment["text"]
        start_time = segment["start"]
        end_time = segment["end"]

        detected_lang, _ = langid.classify(sentence)

        print(f"\n🔹 Detected language for sentence: \"{sentence[:30]}...\" → {detected_lang}")

        toxicity_scores = get_perspective_toxicity(sentence, detected_lang)

        # Check if any attribute exceeds the threshold
        if any(toxicity_scores.get(attr, 0) > 0.4 for attr in
               ["IDENTITY_ATTACK", "SEVERE_TOXICITY", "TOXICITY", "INSULT", "THREAT", "PROFANITY"]):
            print(f"\n🛑 Hate Speech Detected in Sentence: \"{sentence}\" ({start_time:.2f}s - {end_time:.2f}s)")
            for attr in ["IDENTITY_ATTACK", "SEVERE_TOXICITY", "TOXICITY", "INSULT", "THREAT", "PROFANITY"]:
                score = toxicity_scores.get(attr, 0)
                print(f"   {attr}: {score * 100:.2f}%")

            hate_speech_sentences.append((sentence, start_time, end_time))
            word_intervals[sentence] = [(start_time, end_time)]
        else:
            print(f"✅ Clean sentence: \"{sentence[:30]}...\"")

    return hate_speech_sentences, word_intervals




# Allow user to select which hate speech sentences to mute
def select_sentences_to_mute(hate_speech_sentences):
    print("\n🛑 Hate Speech Detected:")
    for idx, (sentence, start, end) in enumerate(hate_speech_sentences, start=1):
        print(f"{idx}. \"{sentence}\" ({start:.2f}s - {end:.2f}s)")

    selected_indices = input("\nEnter the numbers of the sentences you want to mute (comma-separated): ")
    selected_indices = [int(i) - 1 for i in selected_indices.split(",")]

    return [hate_speech_sentences[i] for i in selected_indices]

# Mute inappropriate segments in audio and mux with video
def mute_inappropriate_audio(video_path, selected_sentences, buffer=0.2):
    intervals_to_mute = []
    for sentence, start, end in selected_sentences:
        word_duration = end - start
        dynamic_buffer = 0.3 if word_duration > 0.7 else buffer
        start_buf = max(0, start - dynamic_buffer)
        end_buf = end + dynamic_buffer
        intervals_to_mute.append((start_buf, end_buf))

    if not intervals_to_mute:
        print("✅ No intervals to mute. Skipping audio modification.")
        return None

    # Merge overlapping intervals
    merged_intervals = merge_intervals(intervals_to_mute)
    print("🔹 Merging intervals for muting:")
    for interval in merged_intervals:
        print(f"   Mute from {interval[0]:.2f} to {interval[1]:.2f} seconds")

    filter_chain_parts = [f"volume=enable='between(t,{start:.2f},{end:.2f})':volume=0" for start, end in merged_intervals]
    filter_chain = ",".join(filter_chain_parts)

    temp_audio = "temp_audio.wav"
    muted_audio = "temp_audio_muted.wav"

    print("🔹 Extracting audio from video...")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", temp_audio], check=True)

    print("🔹 Muting inappropriate segments in audio...")
    subprocess.run(["ffmpeg", "-y", "-i", temp_audio, "-af", filter_chain, muted_audio], check=True)

    base, ext = os.path.splitext(video_path)
    output_video = base + "_muted" + ext

    print("🔹 Muxing muted audio with original video...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-i", muted_audio, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map",
         "1:a:0", output_video], check=True)

    os.remove(temp_audio)
    os.remove(muted_audio)
    print(f"✅ Output video saved as {output_video}")
    return output_video

# Merge overlapping mute intervals
def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort()
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Overlapping
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged

# Main function with formatted output
def main(video_path):
    temp_audio = "temp_audio.wav"

    try:
        print("🔹 Extracting audio from video...")
        extract_audio(video_path, temp_audio)

        print("🔹 Transcribing audio with Whisper...")
        transcriptions, _, detected_lang = transcribe_audio(temp_audio)

        print(f"🔹 Detected language: {detected_lang}")

        print("🔹 Analyzing hate speech and profanity...")
        # hate_speech_sentences, word_intervals = analyze_hate_speech(transcriptions, detected_lang)
        hate_speech_sentences, word_intervals = analyze_hate_speech(transcriptions)

        if hate_speech_sentences:
            # Let user select sentences to mute
            selected_sentences = select_sentences_to_mute(hate_speech_sentences)

            print("\n🔹 Muting selected hate speech sentences in the audio...")
            output_video = mute_inappropriate_audio(video_path, selected_sentences, buffer=0.3)

            if output_video:
                print(f"✅ Censored video saved as {output_video}")
        else:
            print("✅ No hate speech detected.")

    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            print("✅ Temporary audio file deleted.")

    print("\n✅ Processing completed successfully!")

if __name__ == "__main__":
    video_file = r"video path"
    main(video_file)
