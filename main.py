import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
import sounddevice as sd
import numpy as np
import threading
import queue
import torch

class PDFReader:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Reader")

        self.text_widget = tk.Text(root, wrap="word", width=80, height=20)
        self.text_widget.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="nsew")
        self.text_widget.config(font=("Helvetica", 12))

        self.open_button = tk.Button(root, text="Open PDF", command=self.open_pdf, width=10)
        self.open_button.grid(row=1, column=0, padx=5, pady=(0, 10), sticky="w")

        self.read_aloud_button = tk.Button(root, text="Read Aloud", command=self.read_aloud, width=10)
        self.read_aloud_button.grid(row=1, column=1, padx=5, pady=(0, 10), sticky="e")

        self.pause_button = tk.Button(root, text="Pause", command=self.pause_reading, width=10)
        self.pause_button.grid(row=2, column=0, padx=5, pady=(0, 10), sticky="w")
        self.pause_button.config(state=tk.DISABLED)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_reading, width=10)
        self.stop_button.grid(row=2, column=1, padx=5, pady=(0, 10), sticky="e")
        self.stop_button.config(state=tk.DISABLED)

        self.initialize_tts()

        self.is_paused = False
        self.stop_requested = False

        self.status_bar = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=3, column=0, columnspan=2, sticky="ew")

        self.chunk_size = 1000  # Increase chunk size to ensure larger inputs
        self.queue = queue.Queue()

    def initialize_tts(self):
        manager = ModelManager()
        try:
            paths = manager.download_model('tts_models/en/ljspeech/speedy-speech')
            # paths = manager.download_model('tts_models/en/ljspeech/tacotron2-DDC')
            print("Model paths:", paths)
            self.model_path = paths[0] if len(paths) > 0 else None
            self.config_path = paths[1] if len(paths) > 1 else None
            self.vocoder_path = None  # Not used since paths[2] is metadata
            self.vocoder_config_path = None  # Not used since paths[2] is metadata

            print("Model path:", self.model_path)
            print("Config path:", self.config_path)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Using device:", self.device)
            self.tts_engine = Synthesizer(
                tts_checkpoint=self.model_path,
                tts_config_path=self.config_path,
                use_cuda=torch.cuda.is_available()
            )
            print("TTS engine initialized successfully.")
        except Exception as e:
            print("Error initializing TTS engine:", e)
            self.tts_engine = None

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
        if self.tts_engine is None:
            self.status_bar.config(text="TTS engine not initialized.")
            return

        content = self.text_widget.get(1.0, tk.END)
        if content.strip():
            self.disable_controls()
            self.status_bar.config(text="Synthesizing and playing...")
            self.stop_requested = False
            threading.Thread(target=self._process_and_play_text, args=(content,)).start()

    def _process_and_play_text(self, text):
        for i in range(0, len(text), self.chunk_size):
            if self.stop_requested:
                break
            chunk = text[i:i+self.chunk_size]
            if len(chunk.strip()) == 0:  # Skip empty chunks
                continue
            try:
                wav = self.tts_engine.tts(chunk)
                wav_np = np.array(wav, dtype=np.float32)
                self.queue.put(wav_np)
            except Exception as e:
                print(f"Error processing chunk: {e}")

        self.queue.put(None)  # Signal the end of the text

        self._play_audio_from_queue()

    def _play_audio_from_queue(self):
        while not self.queue.empty():
            wav_np = self.queue.get()
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

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")
    pdf_reader = PDFReader(root)
    root.mainloop()
