"""
Example usage of the AudioMixer.
"""

from modules.mixer import AudioMixer, MixerConfig, TrackSpec


def main():
    tracks = [
        TrackSpec(path="voice.wav", kind="speech"),
        TrackSpec(path="music.wav", kind="music", gain_db=-3.0),
        TrackSpec(path="bg_ch1.wav", kind="background", channel="ch1", gain_db=-6.0),
    ]

    config = MixerConfig(
        output_sample_rate=48000,
        output_channels=2,
        speech_gain_db=0.0,
        music_gain_db=-4.0,
        background_channel_gains_db={
            'ch1': -6.0,
            'ch2': -8.0,
            'ch3': -8.0,
        },
        normalize=True,
        target_peak_dbfs=-1.0,
    )

    mixer = AudioMixer(config)
    out = mixer.mix(tracks, output_path="mixed.wav")
    print(f"Mixed file saved to: {out}")


if __name__ == "__main__":
    main()


