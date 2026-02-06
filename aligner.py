import os
import re
import torch
import folder_paths
from huggingface_hub import snapshot_download

SUPPORTED_LANGUAGES = [
    "Chinese",
    "English",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
]


class Qwen3ForcedAlignerLoader:
    """加载 Qwen3-ForcedAligner 模型"""
    
    MODELS = {
        "Qwen3-ForcedAligner-0.6B": "Qwen/Qwen3-ForcedAligner-0.6B",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(cls.MODELS.keys()), {"default": "Qwen3-ForcedAligner-0.6B"}),
            },
        }
    
    RETURN_TYPES = ("QWEN3_ALIGNER",)
    RETURN_NAMES = ("aligner",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-ASR"

    def load_model(self, model_name):
        from qwen_asr import Qwen3ForcedAligner
        
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
        
        aligner = Qwen3ForcedAligner.from_pretrained(
            local_model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        
        print(f"[Qwen3-ASR] ForcedAligner loaded: {model_name}")
        return (aligner,)


class Qwen3ForcedAlign:
    """使用 Qwen3-ForcedAligner 进行文本-语音对齐，返回时间戳"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aligner": ("QWEN3_ALIGNER",),
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True}),
                "language": (SUPPORTED_LANGUAGES, {"default": "Chinese"}),
                "segment_by_sentence": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("timestamps", "text_list", "start_times", "end_times",)
    FUNCTION = "align"
    CATEGORY = "Qwen3-ASR"

    def align(self, aligner, audio, text, language, segment_by_sentence=True):
        import numpy as np
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        if waveform.ndim == 3:
            waveform = waveform[0]
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=0)
        
        waveform = waveform.astype(np.float32)
        
        results = aligner.align(
            audio=(waveform, sample_rate),
            text=text,
            language=language,
        )
        
        items = results[0]
        
        texts, starts, ends = [], [], []
        
        if segment_by_sentence:
            segments = [s for s in re.split(r'[。？！，、；.?!,;：:\s]+', text) if s]
            item_idx = 0
            for seg in segments:
                seg_start = None
                seg_end = None
                matched = ""
                while item_idx < len(items) and len(matched) < len(seg):
                    if seg_start is None:
                        seg_start = items[item_idx].start_time
                    seg_end = items[item_idx].end_time
                    matched += items[item_idx].text
                    item_idx += 1
                if seg_start is not None:
                    texts.append(seg)
                    starts.append(f"{seg_start:.3f}")
                    ends.append(f"{seg_end:.3f}")
        else:
            for item in items:
                texts.append(item.text)
                starts.append(f"{item.start_time:.3f}")
                ends.append(f"{item.end_time:.3f}")
        
        timestamps_str = "\n".join(f"{t}\t{s}\t{e}" for t, s, e in zip(texts, starts, ends))
        
        print(f"[Qwen3-ASR] Alignment completed, {len(texts)} segments")
        
        return (timestamps_str, "\n".join(texts), "\n".join(starts), "\n".join(ends),)


NODE_CLASS_MAPPINGS = {
    "Qwen3ForcedAlignerLoader": Qwen3ForcedAlignerLoader,
    "Qwen3ForcedAlign": Qwen3ForcedAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3ForcedAlignerLoader": "Qwen3 ForcedAligner Loader",
    "Qwen3ForcedAlign": "Qwen3 Forced Align",
}
