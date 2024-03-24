import argparse
import functools
import platform
from pyannote.audio import Pipeline
from pyannote_utils import diarize_text

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM

from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path",  type=str,  default="dataset/test.wav", help="input wav file")
add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune/", help="model path")
add_arg("use_gpu",     type=bool, default=True,      help="use GPU?")
add_arg("language",    type=str,  default="chinese", help="language")
add_arg("num_beams",   type=int,  default=1,         help="beam search size")
add_arg("batch_size",  type=int,  default=16,        help="batch_size for inference")
add_arg("use_compile", type=bool, default=False,     help="use Pytorch2.0 compiler?")
add_arg("task",        type=str,  default="transcribe", choices=['transcribe', 'translate'], help="task of model, transcribe or translate")
add_arg("assistant_model_path",  type=str,  default=None,  help="assistant mode path, for example openai/whisper-tiny")
add_arg("local_files_only",      type=bool, default=True,  help="local file only?")
add_arg("use_flash_attention_2", type=bool, default=False, help="use FlashAttention2 to speedup?")
add_arg("use_bettertransformer", type=bool, default=False, help="use BetterTransformer to speedup?")
args = parser.parse_args()
print_arguments(args)

device = "cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() and args.use_gpu else torch.float32

processor = AutoProcessor.from_pretrained(args.model_path)

# get model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    args.model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    use_flash_attention_2=args.use_flash_attention_2
)
if args.use_bettertransformer and not args.use_flash_attention_2:
    model = model.to_bettertransformer()

# use Pytorch2.0 compiler?
if args.use_compile:
    if torch.__version__ >= "2" and platform.system().lower() != 'windows':
        model = torch.compile(model)
model.to(device)

# get assistant model
generate_kwargs_pipeline = None
if args.assistant_model_path is not None:
    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    assistant_model.to(device)
    generate_kwargs_pipeline = {"assistant_model": assistant_model}

# create pipe
infer_pipe = pipeline("automatic-speech-recognition",
                      model=model,
                      tokenizer=processor.tokenizer,
                      feature_extractor=processor.feature_extractor,
                      max_new_tokens=128,
                      chunk_length_s=30,
                      batch_size=args.batch_size,
                      torch_dtype=torch_dtype,
                      generate_kwargs=generate_kwargs_pipeline,
                      device=device)

# language
generate_kwargs = {"task": args.task, "num_beams": args.num_beams}
if args.language is not None:
    generate_kwargs["language"] = args.language
# inference
asr_result = infer_pipe(args.audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)

#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
#                                    use_auth_token="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")
#diarization_result = pipeline(args.audio_path)
#final_result = diarize_text(asr_result["chunks"], diarization_result)

#print(diarization_result)

#for seg in final_result:
#    print(seg[0], seg[1], seg[2])



for chunk in asr_result["chunks"]:
    print(f"{chunk['text']}", end='')

#for chunk in asr_result["chunks"]:
#    print(f"[{chunk['timestamp'][0]}-{chunk['timestamp'][1]}s] {chunk['text']}")

