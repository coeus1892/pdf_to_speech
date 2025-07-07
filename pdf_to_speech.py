#!/usr/bin/env python3
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
from PyPDF2 import PdfReader
import re
import sys
import time
import warnings

class PDFToSpeechConverter:
    def __init__(self, device="cpu", voice_prompt=None):
        """Initialize with CPU (Mac-compatible) and optional voice prompt"""
        self.device = device
        self.voice_prompt = Path(voice_prompt) if voice_prompt else None
        self.model = None
        self.setup_dirs()

    def setup_dirs(self):
        """Create required directories"""
        Path("documents").mkdir(exist_ok=True)
        Path("audio_output").mkdir(exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
            
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?\'":;()\-]', '', text)
        return text.strip()

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            print(f"📖 Opening PDF: {pdf_path.name}")
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            if total_pages == 0:
                raise ValueError("PDF contains no pages")
                
            full_text = []
            for i, page in enumerate(reader.pages, 1):
                print(f"  Processing page {i}/{total_pages}...")
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = self.clean_text(page_text)
                        full_text.append(cleaned_text)
                except Exception as e:
                    warnings.warn(f"Could not extract text from page {i}: {e}")
                    
            if not full_text:
                raise RuntimeError("No text could be extracted from any page")
                
            return "\n".join(full_text)
            
        except Exception as e:
            raise RuntimeError(f"❌ PDF processing failed: {e}")

    def initialize_model(self):
        """Lazy-load TTS model to save memory"""
        if self.model is None:
            print("🔈 Loading TTS model...")
            start_time = time.time()
            try:
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                load_time = time.time() - start_time
                print(f"Model loaded in {load_time:.1f} seconds")
            except Exception as e:
                raise RuntimeError(f"Failed to load TTS model: {e}")

    def text_to_speech(self, text: str, output_path: Path):
        """Convert text to speech with progress feedback"""
        if not text.strip():
            raise ValueError("No text to convert to speech")
            
        self.initialize_model()
        output_path.parent.mkdir(exist_ok=True, parents=True)

        print(f"🎙️ Converting {len(text)} characters to speech...")
        try:
            kwargs = {"audio_prompt_path": str(self.voice_prompt)} if self.voice_prompt else {}
            wav = self.model.generate(text, **kwargs)
            ta.save(output_path, wav, self.model.sr)
            print(f" Success! Audio saved to:\n{output_path.absolute()}")
        except Exception as e:
            raise RuntimeError(f" Speech generation failed: {e}")

def main():

    print("hi")
    try:
        converter = PDFToSpeechConverter(
            device="cpu", 
            voice_prompt=None 
        )

        filename = input(" Enter PDF filename: ").strip()
        if not filename:
            print("No filename provided.")
            return
            
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
            
        pdf_path = Path("documents") / filename
        output_path = Path("audio_output") / f"{pdf_path.stem}.wav"
        text = converter.extract_text_from_pdf(pdf_path)
        
        print("\n Text being converted to speech:\n" + "="*50)
        print(text[:1000] + ("..." if len(text) > 2000 else ""))  
        print("="*50 + "\n")
        
        converter.text_to_speech(text, output_path)
        
        print("\n Conversion complete! ")

    except Exception as e:
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())