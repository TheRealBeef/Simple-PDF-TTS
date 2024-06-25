import fitz
import tkinter as tk
from tkinter import filedialog
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
import sounddevice as sd

class PDFReader:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Reader")

        self.text_widget = tk.Text(root, wrap="word", width=80, height=20)
        self.text_widget.grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="nsew")
        self.text_widget.config(font=("Helvetica", 12))

        self.open_button = tk.Button(root, text="Open PDF", command=self.open_pdf, width=10)
        self.open_button.grid(row=1, column=0, padx=5, pady=(0,10), sticky="w")

        self.read_aloud_button = tk.Button(root, text="Read Aloud", command=self.read_aloud, width=10)
        self.read_aloud_button.grid(row=1, column=1, padx=5, pady=(0,10), sticky="e")

        self.pause_button = tk.Button(root, text="Pause", command=self.pause_reading, width=10)
        self.pause_button.grid(row=2, column=0, padx=5, pady=(0,10), sticky="w")
        self.pause_button.config(state=tk.DISABLED)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_reading, width=10)
        self.stop_button.grid(row=2, column=1, padx=5, pady=(0,10), sticky="e")
        self.stop_button.config(state=tk.DISABLED)

        # Initialize TTS
        manager = ModelManager()
        self.model_path, self.config_path, self.speaker_path, self.vocoder_path, self.vocoder_config_path = manager.download_model('tts_models/en/ljspeech/tacotron2-DDC')
        self.tts_engine = Synthesizer(self.model_path, self.config_path, self.speaker_path, self.vocoder_path, self.vocoder_config_path)

        self.is_paused = False

        self.status_bar = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=3, column=0, columnspan=2, sticky="ew")

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
        content = self.text_widget.get(1.0, tk.END)
        if content.strip():
            try:
                self.status_bar.config(text="Reading aloud...")
                self.disable_controls()
                wav = self.tts_engine.tts(content)
                sd.play(wav)
                sd.wait()
                self.status_bar.config(text="Reading completed.")
                self.enable_controls()
            except Exception as e:
                self.status_bar.config(text=f"Error: {str(e)}")

    def pause_reading(self):
        if not self.is_paused:
            self.is_paused = True
            sd.stop()
            self.status_bar.config(text="Paused.")
            self.pause_button.config(text="Resume")
        else:
            self.is_paused = False
            self.read_aloud()
            self.status_bar.config(text="Reading resumed.")
            self.pause_button.config(text="Pause")

    def stop_reading(self):
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
