import os
import numpy as np
import torch
import torchaudio
import folder_paths
from huggingface_hub import snapshot_download


class Qwen3ASRLoader:
    """加载 Qwen3-ASR 模型"""
    
    MODELS = {
        "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
        "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(cls.MODELS.keys()), {"default": "Qwen3-ASR-1.7B"}),
            },
        }
    
    RETURN_TYPES = ("QWEN3ASR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-ASR"

    def load_model(self, model_name):
        from qwen_asr import Qwen3ASRModel
        
        models_dir = folder_paths.models_dir
        local_model_path = os.path.join(models_dir, "Qwen3-ASR", model_name)
        
        if not os.path.exists(local_model_path) or not os.listdir(local_model_path):
            print(f"[Qwen3-ASR] Model not found, downloading: {model_name}")
            os.makedirs(local_model_path, exist_ok=True)
            hf_repo = self.MODELS[model_name]
            snapshot_download(
                repo_id=hf_repo,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
            )
            print(f"[Qwen3-ASR] Model downloaded: {local_model_path}")
        else:
            print(f"[Qwen3-ASR] Using local model: {local_model_path}")
        
        model = Qwen3ASRModel.from_pretrained(
            local_model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=32,
            max_new_tokens=512,
        )
        
        print(f"[Qwen3-ASR] Model loaded: {model_name}")
        return (model,)


SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]


class Qwen3ASRTranscribe:
    """使用 Qwen3-ASR 模型进行语音转文字"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3ASR_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {"default": "Auto"}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("text", "language",)
    FUNCTION = "transcribe"
    CATEGORY = "Qwen3-ASR"

    def transcribe(self, model, audio, language="Auto"):
        # 处理 ComfyUI AUDIO 格式: {"waveform": tensor, "sample_rate": int}
        waveform = audio["waveform"]  # shape: (batch, channels, samples)
        sample_rate = audio["sample_rate"]
        
        # 转换为 numpy，取第一个 batch，转为单声道
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # 处理维度: (batch, channels, samples) -> (samples,)
        if waveform.ndim == 3:
            waveform = waveform[0]  # 取第一个 batch
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=0)  # 多声道转单声道
        
        waveform = waveform.astype(np.float32)
        
        # 确定语言参数
        lang_param = None if language == "Auto" else language
        
        # 调用模型转录
        results = model.transcribe(
            audio=(waveform, sample_rate),
            language=lang_param,
            return_time_stamps=False,
        )
        
        result = results[0]
        text = result.text
        detected_language = result.language
        
        print(f"[Qwen3-ASR] Detected language: {detected_language}")
        print(f"[Qwen3-ASR] Transcribed text: {text}")
        
        return (text, detected_language,)


NODE_CLASS_MAPPINGS = {
    "Qwen3ASRLoader": Qwen3ASRLoader,
    "Qwen3ASRTranscribe": Qwen3ASRTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ASRLoader": "Qwen3-ASR Loader",
    "Qwen3ASRTranscribe": "Qwen3-ASR Transcribe",
}
