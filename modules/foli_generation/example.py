"""
Example usage of the Foley generator based on AudioLDM2.
"""

from modules.foli_generation import FoliGenerator, FoliGenConfig


def main():
    config = FoliGenConfig(
        repo_id="cvssp/audioldm2",
        audio_length_in_s=6.0,
        num_inference_steps=200,
        num_waveforms_per_prompt=1,
    )

    generator = FoliGenerator(config)
    prompt = "The sound of a hammer hitting a wooden surface"
    audio = generator.generate(prompt, seed=0)

    generator.save_audio(audio, "foli_example.wav")
    print("Saved to foli_example.wav")


if __name__ == "__main__":
    main()


