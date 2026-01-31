# ComfyUI SynVow Qwen3-ASR

基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 ComfyUI 语音识别插件。

## 原项目

[https://github.com/QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)

## 功能特性

- 🎤 **语音转文字**：高精度语音识别
- 🌍 **多语言支持**：支持52种语言/方言，自动语言检测
- ⏱️ **强制对齐**：生成字/词级别时间戳
- 🤖 **自动下载模型**：首次使用时自动从 HuggingFace 下载模型
- 📊 **长音频支持**：ASR 最长支持20分钟，对齐最长支持3分钟（超长音频自动分段处理）

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install qwen-asr huggingface_hub torchaudio
```

### 2. 模型下载

首次使用时模型会**自动下载**到以下目录：

```
ComfyUI/models/Qwen3-ASR/
├── Qwen3-ASR-1.7B/           # ASR 模型 (1.7B, 默认)
├── Qwen3-ASR-0.6B/           # ASR 模型 (0.6B, 更轻量)
└── Qwen3-ForcedAligner-0.6B/ # 强制对齐模型
```

## 节点说明

### 1. Qwen3-ASR Loader

加载 Qwen3-ASR 语音识别模型。

**输入参数：**
- `model_name`：选择模型版本
  - `Qwen3-ASR-1.7B`（默认）- 精度更高
  - `Qwen3-ASR-0.6B` - 速度更快，显存占用更低

**输出：**
- `model`：ASR 模型，用于转录节点

### 2. Qwen3-ASR Transcribe

语音转文字。

**输入参数：**
- `model`：来自 Loader 节点的模型
- `audio`：音频输入（ComfyUI AUDIO 类型）
- `language`：语言选择（默认：Auto 自动检测）

**输出：**
- `text`：识别的文本
- `language`：检测到的语言

### 3. Qwen3 ForcedAligner Loader

加载强制对齐模型，用于生成时间戳。

**输入参数：**
- `model_name`：选择模型版本
  - `Qwen3-ForcedAligner-0.6B`（默认）

**输出：**
- `aligner`：对齐器模型，用于对齐节点

### 4. Qwen3 Forced Align

生成字/词级别时间戳。

**输入参数：**
- `aligner`：来自 Loader 节点的对齐器
- `audio`：音频输入（ComfyUI AUDIO 类型）
- `text`：要对齐的文本
- `language`：语言（支持11种语言）

**输出：**
- `timestamps`：时间戳，格式为 `文字\t开始时间\t结束时间`

## 使用示例

### 基础语音转文字

1. 使用 `Load Audio` 节点加载音频文件
2. 连接 `Qwen3-ASR Loader` 加载模型
3. 连接 `Qwen3-ASR Transcribe` 进行转录
4. 输出文本和检测到的语言

### 带时间戳

1. `Load Audio` → `Qwen3-ASR Loader` → `Qwen3-ASR Transcribe` → 获取文本
2. `Qwen3 ForcedAligner Loader` → `Qwen3 Forced Align`（输入音频+文本）→ 获取时间戳

## 支持的语言

**ASR（52种语言）：** 中文、英语、粤语、阿拉伯语、德语、法语、西班牙语、葡萄牙语、印尼语、意大利语、韩语、俄语、泰语、越南语、日语、土耳其语、印地语、马来语、荷兰语、瑞典语、丹麦语、芬兰语、波兰语、捷克语、菲律宾语、波斯语、希腊语、罗马尼亚语、匈牙利语、马其顿语等。

**强制对齐（11种语言）：** 中文、英语、粤语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语

## 注意事项

1. **显存需求**：
   - Qwen3-ASR-0.6B：约 4GB 显存
   - Qwen3-ASR-1.7B：约 8GB 显存
   - ForcedAligner-0.6B：约 4GB 显存

2. **音频长度限制**：
   - ASR：最长 20 分钟（1200秒）
   - 强制对齐：最长 3 分钟（180秒）
   - 超长音频会自动分段处理后合并

## 许可证

本项目遵循原 Qwen3-ASR 项目的许可证。

## 相关链接

- [Qwen3-ASR 原项目](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR HuggingFace 模型](https://huggingface.co/Qwen)
