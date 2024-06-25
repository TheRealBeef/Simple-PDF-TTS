import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog, ttk
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading
import queue
import torch
import re
import librosa

class PDFReader:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Reader")

        try:
            all_models = TTS().list_models().list_models()
            self.models = [model for model in all_models if '/en/' in model or '/multilingual/' in model]
        except Exception as e:
            self.models = []
            print(f"Error retrieving TTS models: {e}")

        # Creating UI elements
        self.text_widget = tk.Text(root, wrap="word", width=80, height=20, undo=True)
        self.text_widget.grid(row=0, column=0, columnspan=4, padx=10, pady=(10, 5), sticky="nsew")
        self.text_widget.config(font=("Helvetica", 12))
        self.text_widget.bind("<Control-a>", self.select_all)

        self.scrollbar = tk.Scrollbar(root, command=self.text_widget.yview)
        self.scrollbar.grid(row=0, column=4, sticky='ns')
        self.text_widget['yscrollcommand'] = self.scrollbar.set

        self.model_label = tk.Label(root, text="Select TTS Model:")
        self.model_label.grid(row=1, column=0, padx=5, pady=(0, 10), sticky="w")

        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(root, textvariable=self.model_var, values=self.models, width=60)
        self.model_menu.grid(row=1, column=1, padx=5, pady=(0, 10), sticky="w")
        if self.models:
            self.model_menu.current(0)

        self.speed_label = tk.Label(root, text="Playback Speed:")
        self.speed_label.grid(row=1, column=2, padx=5, pady=(0, 10), sticky="w")

        self.speed_slider = tk.Scale(root, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
        self.speed_slider.set(1.0)
        self.speed_slider.grid(row=1, column=3, padx=5, pady=(0, 10), sticky="w")

        self.open_button = tk.Button(root, text="Open PDF", command=self.open_pdf, width=10)
        self.open_button.grid(row=2, column=0, padx=5, pady=(0, 10), sticky="w")

        self.read_aloud_button = tk.Button(root, text="Read Aloud", command=self.read_aloud, width=10)
        self.read_aloud_button.grid(row=2, column=1, padx=5, pady=(0, 10), sticky="e")

        self.pause_button = tk.Button(root, text="Pause", command=self.pause_reading, width=10)
        self.pause_button.grid(row=3, column=0, padx=5, pady=(0, 10), sticky="w")
        self.pause_button.config(state=tk.DISABLED)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_reading, width=10)
        self.stop_button.grid(row=3, column=1, padx=5, pady=(0, 10), sticky="e")
        self.stop_button.config(state=tk.DISABLED)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = None

        self.is_paused = False
        self.stop_requested = False

        self.status_bar = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=4, column=0, columnspan=4, sticky="ew")

        self.chunk_size = 2000  # Target chunk size for TTS processing
        self.min_length = 300  # Minimum length to avoid kernel size issues
        self.queue = queue.PriorityQueue()  # Use a priority queue to maintain order

    def initialize_tts(self):
        model_name = self.model_var.get()
        if not model_name:
            self.status_bar.config(text="Please select a TTS model.")
            return False
        try:
            self.tts = TTS(model_name=model_name, progress_bar=False).to(self.device)
            print("TTS engine initialized successfully.")
            return True
        except Exception as e:
            print("Error initializing TTS engine:", e)
            self.status_bar.config(text=f"Error initializing TTS engine: {e}")
            self.tts = None
            return False

    def open_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            try:
                pdf_document = fitz.open(file_path)
                num_pages = pdf_document.page_count

                self.status_bar.config(text=f"Loading {num_pages} pages...")
                self.root.update_idletasks()

                content = ""
                for page_number in range(num_pages):
                    page = pdf_document.load_page(page_number)
                    content += page.get_text()

                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(tk.END, content)
                self.status_bar.config(text=f"PDF loaded successfully.")
                self.enable_controls()
            except Exception as e:
                self.status_bar.config(text=f"Error: {str(e)}")

    def read_aloud(self):
        if not self.initialize_tts():
            return

        content = self.text_widget.get(1.0, tk.END)
        if content.strip():
            self.disable_controls()
            self.status_bar.config(text="Synthesizing and playing...")
            self.stop_requested = False
            threading.Thread(target=self._process_and_play_text, args=(content,)).start()

    def _process_and_play_text(self, text):
        chunks = self._split_into_chunks(text)
        for i, chunk in enumerate(chunks):
            if self.stop_requested:
                break
            chunk = self._clean_text(chunk.strip())
            try:
                if len(chunk) < self.min_length:
                    print(f"Skipping chunk {i} due to insufficient length: {len(chunk)} characters.")
                    continue
                self.highlight_text(chunk, i)
                print(f"Processing chunk {i} with length {len(chunk)}: {chunk[:50]}...")
                wav = self.tts.tts(chunk)
                wav_np = np.array(wav, dtype=np.float32)
                speed_factor = self.speed_slider.get()
                wav_np = librosa.effects.time_stretch(wav_np.astype(np.float32), rate=speed_factor)
                wav_np = librosa.effects.pitch_shift(wav_np, sr=22050, n_steps=(speed_factor - 1) * 12)
                sd.play(wav_np, samplerate=22050)
                sd.wait()
                self.remove_highlight(i)
                self.queue.put((i, wav_np))  # Use a tuple to maintain the order
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")

        self.queue.put((float('inf'), None))  # Signal the end of the text

    def highlight_text(self, chunk, chunk_index):
        # Highlight the current chunk being read
        start_index = self.text_widget.search(chunk[:50], "1.0", tk.END)
        if not start_index:
            print(f"Chunk {chunk_index} not found in text widget.")
            return
        end_index = f"{start_index}+{len(chunk)}c"
        self.text_widget.tag_add(f"highlight{chunk_index}", start_index, end_index)
        self.text_widget.tag_config(f"highlight{chunk_index}", background="yellow")
        self.root.update_idletasks()

    def remove_highlight(self, chunk_index):
        # Remove the highlight from the previously read chunk
        self.text_widget.tag_delete(f"highlight{chunk_index}")
        self.root.update_idletasks()

    def _split_into_chunks(self, text):
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if len(current_chunk.strip()) >= self.min_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            current_chunk += sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _clean_text(self, text):
        # Remove unsupported characters, newline characters, parentheses, hyphens, brackets, and quotes
        text = text.replace('\n', ' ')
        text = re.sub(r'[\(\)\[\]\-\"\']', '', text)
        return re.sub(r'[^A-Za-z0-9 .,!?]', '', text)

    def _play_audio_from_queue(self):
        while not self.queue.empty():
            priority, wav_np = self.queue.get()
            if wav_np is None:
                break
            sd.play(wav_np, samplerate=22050)
            sd.wait()

        self.status_bar.config(text="Reading completed.")
        self.enable_controls()

    def pause_reading(self):
        if not self.is_paused:
            self.is_paused = True
            sd.stop()
            self.status_bar.config(text="Paused.")
            self.pause_button.config(text="Resume")
        else:
            self.is_paused = False
            self._play_audio_from_queue()
            self.status_bar.config(text="Reading resumed.")
            self.pause_button.config(text="Pause")

    def stop_reading(self):
        self.stop_requested = True
        sd.stop()
        self.status_bar.config(text="Reading stopped.")
        self.enable_controls()

    def disable_controls(self):
        self.open_button.config(state=tk.DISABLED)
        self.read_aloud_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)

    def enable_controls(self):
        self.open_button.config(state=tk.NORMAL)
        self.read_aloud_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)

    def select_all(self, event=None):
        self.text_widget.tag_add(tk.SEL, "1.0", tk.END)
        return "break"

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1920x1080")
    pdf_reader = PDFReader(root)
    root.mainloop()
