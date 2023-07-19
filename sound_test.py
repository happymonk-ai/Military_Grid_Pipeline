import pydsm
import time

def play_audio(file_path):
    pulse = pydsm.PulseAudio()
    pulse.play(file_path)

    # Wait for the audio to finish playing
    while pulse.is_playing():
        time.sleep(0.1)

    # Clean up the PulseAudio resources
    pulse.close()

# Usage example
audio_file_path = './alarm.wav'
play_audio(audio_file_path)


