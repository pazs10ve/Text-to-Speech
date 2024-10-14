import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

default_description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
last_description = default_description

def generate_and_play():
    global last_description

    user_description = description_input.get("1.0", tk.END).strip()
    user_text = text_input.get("1.0", tk.END).strip()

    if not user_text:
        messagebox.showwarning("Input Required", "Please enter some text.")
        return

    if user_description:
        last_description = user_description
    else:
        print(f"Using last description: {last_description}")

    try:
        print("Generating speech...")

        input_ids = tokenizer(last_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(user_text, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        print("Playing the audio...")
        sd.play(audio_arr, samplerate=model.config.sampling_rate)
        sd.wait()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error: {e}")

app = tk.Tk()
app.title("Text-to-Speech Generator")

tk.Label(app, text="Speaker Description:").pack(pady=5)
description_input = tk.Text(app, height=4, width=50)
description_input.pack()

tk.Label(app, text="Enter Text:").pack(pady=5)
text_input = tk.Text(app, height=4, width=50)
text_input.pack()

submit_button = tk.Button(app, text="Generate and Play", command=generate_and_play)
submit_button.pack(pady=20)
app.mainloop()
