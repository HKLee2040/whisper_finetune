import argparse
import functools
import os

from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast,\
    WhisperProcessor
from peft import PeftModel, PeftConfig
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("lora_model", type=str, default="output/whisper-tiny/checkpoint-best/", help="check point path")
add_arg('output_dir', type=str, default='models/',    help="path for merge mode")
add_arg("local_files_only", type=bool, default=False, help="local model")
args = parser.parse_args()
print_arguments(args)

# check the exist of model
assert os.path.exists(args.lora_model), f"model {args.lora_model} not exists"
# get Lora configuration
peft_config = PeftConfig.from_pretrained(args.lora_model)
# get Whisper base model
base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, device_map={"": "cpu"},
                                                             local_files_only=args.local_files_only)
# merge Lora model with whisper based model
model = PeftModel.from_pretrained(base_model, args.lora_model, local_files_only=args.local_files_only)
feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path,
                                                            local_files_only=args.local_files_only)
tokenizer = WhisperTokenizerFast.from_pretrained(peft_config.base_model_name_or_path,
                                                 local_files_only=args.local_files_only)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path,
                                             local_files_only=args.local_files_only)

# merge parameters
model = model.merge_and_unload()
model.train(False)

if peft_config.base_model_name_or_path.endswith("/"):
    peft_config.base_model_name_or_path = peft_config.base_model_name_or_path[:-1]
save_directory = os.path.join(args.output_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
os.makedirs(save_directory, exist_ok=True)

# save model to the path
model.save_pretrained(save_directory, max_shard_size='4GB')
feature_extractor.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f'合并模型保持在：{save_directory}')
