import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import sounddevice as sd

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    default_description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    last_description = default_description

    while True:
        print(f"Enter speaker description (press Enter to use last description or default):")
        description_input = "\n".join(iter(input, ''))
        if description_input.strip():
            last_description = description_input
        else:
            print(f"Using last description: {last_description}")

        print("Enter text prompt (or type 'exit' to quit):")
        prompt = "\n".join(iter(input, ''))
        if prompt.lower().strip() == 'exit':
            print("Exiting...")
            break

        input_ids = tokenizer(last_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        print("Generating speech...")
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        print("Playing the audio...")
        sd.play(audio_arr, samplerate=model.config.sampling_rate)
        sd.wait()

if __name__ == "__main__":
    main()
