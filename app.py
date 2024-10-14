import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import sounddevice as sd
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import time
import threading

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
        current_time = time.strftime("%H:%M:%S", time.localtime())
        chat_log.insert(tk.END, f"You ({current_time}): {user_text}\n")

        print("Generating speech...")
        input_ids = tokenizer(last_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(user_text, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        audio_length = len(audio_arr) / model.config.sampling_rate
        chat_log.insert(tk.END, f"Generated audio length: {audio_length:.2f} seconds\n")
        chat_log.see(tk.END)

        progress_bar["maximum"] = audio_length
        threading.Thread(target=play_audio_and_update_progress, args=(audio_arr, audio_length)).start()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error: {e}")

def play_audio_and_update_progress(audio_arr, audio_length):
    try:
        sd.play(audio_arr, samplerate=model.config.sampling_rate)
        for t in range(int(audio_length)):
            progress_bar["value"] = t
            time.sleep(1)
        sd.wait()
        progress_bar["value"] = 0 
    except Exception as e:
        messagebox.showerror("Playback Error", f"Error playing audio: {e}")
        print(f"Error: {e}")

app = tk.Tk()
app.title("ChatGPT-Like Text-to-Speech Generator")
app.state('zoomed') 

chat_frame = tk.Frame(app)
chat_log = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=20, state=tk.NORMAL)
chat_log.pack(padx=10, pady=10)
chat_frame.pack(padx=10, pady=10)

tk.Label(app, text="Speaker Description:").pack(pady=5)
description_input = tk.Text(app, height=4, width=100)
description_input.pack()

tk.Label(app, text="Enter Text:").pack(pady=5)
text_input = tk.Text(app, height=4, width=100)
text_input.pack()

progress_bar = ttk.Progressbar(app, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

submit_button = tk.Button(app, text="Generate and Play", command=generate_and_play)
submit_button.pack(pady=20)

app.mainloop()
