# ComfyUI SynVow Qwen3-ASR
<img width="2386" height="771" alt="image" src="https://github.com/user-attachments/assets/6f4bef09-b632-46bd-ae71-eb557605d759" />

A ComfyUI speech recognition plugin based on [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).

## Original Project

[https://github.com/QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)

## Features

- � **Speech-to-Text**: Transcribe audio to text with high accuracy
- 🌍 **Multi-language Support**: Supports 52 languages/dialects with automatic language detection
- ⏱️ **Forced Alignment**: Generate word/character-level timestamps
- 🤖 **Auto Model Download**: Automatically download models from HuggingFace on first use
- 📊 **Long Audio Support**: ASR supports up to 20 minutes, alignment supports up to 3 minutes (auto-chunking for longer audio)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install qwen-asr huggingface_hub torchaudio
```

### 2. Model Download

Models will be **automatically downloaded** to the following directory on first use:

```
ComfyUI/models/Qwen3-ASR/
├── Qwen3-ASR-1.7B/           # ASR model (1.7B, default)
├── Qwen3-ASR-0.6B/           # ASR model (0.6B, lighter)
└── Qwen3-ForcedAligner-0.6B/ # Forced alignment model
```

## Node Description

### 1. Qwen3-ASR Loader

Load Qwen3-ASR speech recognition model.

**Input Parameters:**
- `model_name`: Select model version
  - `Qwen3-ASR-1.7B` (default) - Better accuracy
  - `Qwen3-ASR-0.6B` - Faster, lower VRAM

**Output:**
- `model`: ASR model for transcription node

### 2. Qwen3-ASR Transcribe

Transcribe audio to text.

**Input Parameters:**
- `model`: Model from Loader node
- `audio`: Audio input (ComfyUI AUDIO type)
- `language`: Language selection (default: Auto)

**Output:**
- `text`: Transcribed text
- `language`: Detected language

### 3. Qwen3 ForcedAligner Loader

Load forced alignment model for generating timestamps.

**Input Parameters:**
- `model_name`: Select model version
  - `Qwen3-ForcedAligner-0.6B` (default)

**Output:**
- `aligner`: Aligner model for alignment node

### 4. Qwen3 Forced Align

Generate word/character-level timestamps.

**Input Parameters:**
- `aligner`: Aligner from Loader node
- `audio`: Audio input (ComfyUI AUDIO type)
- `text`: Text to align
- `language`: Language (supports 11 languages)

**Output:**
- `timestamps`: Timestamps in format `text\tstart_time\tend_time`

## Usage Example

### Basic Speech-to-Text

1. Use `Load Audio` node to load audio file
2. Connect `Qwen3-ASR Loader` to load model
3. Connect `Qwen3-ASR Transcribe` for transcription
4. Output text and detected language

### With Timestamps

1. `Load Audio` → `Qwen3-ASR Loader` → `Qwen3-ASR Transcribe` → Get text
2. `Qwen3 ForcedAligner Loader` → `Qwen3 Forced Align` (input audio + text) → Get timestamps

## Supported Languages

**ASR (52 languages):** Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian, etc.

**Forced Alignment (11 languages):** Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish

## Notes

1. **VRAM Requirements**: 
   - Qwen3-ASR-0.6B: ~4GB VRAM
   - Qwen3-ASR-1.7B: ~8GB VRAM
   - ForcedAligner-0.6B: ~4GB VRAM

2. **Audio Length Limits**:
   - ASR: Max 20 minutes (1200 seconds)
   - Forced Alignment: Max 3 minutes (180 seconds)
   - Longer audio is automatically chunked and merged

## License

This project follows the license of the original Qwen3-ASR project.

## Related Links

- [Qwen3-ASR Original Project](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR Models on HuggingFace](https://huggingface.co/Qwen)

