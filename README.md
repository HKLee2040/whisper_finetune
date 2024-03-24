# 微調Whisper 語音識別模型

![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/Whisper-Finetune)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/Whisper-Finetune)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/Whisper-Finetune)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

## 支持模型

 - openai/whisper-tiny
 - openai/whisper-base
 - openai/whisper-small
 - openai/whisper-medium
 - openai/whisper-large
 - openai/whisper-large-v2
 - openai/whisper-large-v3


**使用環境：**

- Anaconda 3
- Python 3.8
- Pytorch 2.2.1
- Ubuntu 22.04
- GPU 3090


1. `aishell.py`：製作 AIShell 訓練數據
2. `finetune.py`：微調模型
3. `merge_lora.py`：合併 Whisper 和 Lora 的模型。
4. `evaluation.py`：評估模型
5. `infer.py`：使用模型預測特定音檔。

## 數據準備
<a name='準備數據'></a>

執行 `aishell.py`

**小提示：**
可以自行將訓練資料集轉換成繁體中文


<a name='微調模型'></a>

## 微調模型

```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model=openai/whisper-large-v2 --output_dir=output/
```
在 3090 上訓練較大的模型, 如 whisper-large-v2, 需使用 int8 mode

## 合併模型

微調完成之後會有兩個模型，Whisper基礎模型以及Lora模型，需要把兩個模型合併後才能繼續操作

```shell
python merge_lora.py --lora_model=output/whisper-tiny/checkpoint-best/ --output_dir=models/
```

## 評估模型
<a name='評估模型'></a>


```shell
python evaluation.py --model_path=models/whisper-large-v2-finetune --metric=cer
```

## 預測
<a name='预测'></a>

建議可以先用 Silero VAD 將音檔處理過, 可以有效地降低重複字詞的發生
```shell
python infer.py --audio_path=dataset/test.wav --model_path=models/whisper-large-v2-finetune
```

## 参考资料

1. https://github.com/yeyupiaoling/Whisper-Finetune
