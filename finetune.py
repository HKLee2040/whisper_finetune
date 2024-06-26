import argparse
import functools
import os

from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="dataset/train.json",       help="training dataset path")
add_arg("test_data",     type=str, default="dataset/test.json",        help="test dataset path")
add_arg("base_model",    type=str, default="openai/whisper-tiny",      help="Whisper base model")
add_arg("output_dir",    type=str, default="output/",                  help="trained model path")
add_arg("warmup_steps",  type=int, default=50,      help="warmup steps")
add_arg("logging_steps", type=int, default=100,     help="logging steps")
add_arg("eval_steps",    type=int, default=5000,    help="evaluation steps")
add_arg("save_steps",    type=int, default=5000,    help="save steps")
add_arg("num_workers",   type=int, default=8,       help="number of works")
add_arg("learning_rate", type=float, default=1e-3,  help="learning rate")
add_arg("min_audio_len", type=float, default=0.5,   help="min audio length, unit: second")
add_arg("max_audio_len", type=float, default=30,    help="max audio length, unit: second")
add_arg("use_adalora",   type=bool,  default=True,  help="use adalora or not?")
add_arg("fp16",          type=bool,  default=True,  help="use fp16 model")
add_arg("use_8bit",      type=bool,  default=True, help="quantized int8 model?")
add_arg("timestamps",    type=bool,  default=False, help="use time stamps?")
add_arg("use_compile",   type=bool, default=False, help="use Pytorch2.0 compiler")
add_arg("local_files_only", type=bool, default=False, help="local files only?")
add_arg("num_train_epochs", type=int, default=3,      help="number of training epochs")
add_arg("language",      type=str, default="Chinese", help="language，if None then multi-language")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="tasks")
add_arg("augment_config_path",         type=str, default=None, help="augment config path")
add_arg("resume_from_checkpoint",      type=str, default=None, help="resume from checkpoint")
add_arg("per_device_train_batch_size", type=int, default=8,    help="batch size for training")
add_arg("per_device_eval_batch_size",  type=int, default=8,    help="batch size for evaluation")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="gradient accumulation steps")
args = parser.parse_args()
print_arguments(args)


processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

# load data
train_dataset = CustomDataset(data_list_path=args.train_data,
                              processor=processor,
                              language=args.language,
                              timestamps=args.timestamps,
                              min_duration=args.min_audio_len,
                              max_duration=args.max_audio_len,
                              augment_config_path=args.augment_config_path)
test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             language=args.language,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"training dataset size：{len(train_dataset)}, test dataset size：{len(test_dataset)}")

# padding
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

# get model
model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                        load_in_8bit=args.use_8bit,
                                                        device_map=device_map,
                                                        local_files_only=args.local_files_only)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# quantized model
model = prepare_model_for_kbit_training(model)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

print('config LoRA module...')
if args.resume_from_checkpoint:
    print("Loading adapters from checkpoint.")
    model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
else:
    print(f'adding LoRA modules...')
    target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    print(target_modules)
    if args.use_adalora:
        config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                               lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules)
    else:
        config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)

if args.base_model.endswith("/"):
    args.base_model = args.base_model[:-1]
output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model))

# define training parameters
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,  # output path
                             per_device_train_batch_size=args.per_device_train_batch_size,  # training batch_size
                             per_device_eval_batch_size=args.per_device_eval_batch_size,  # evaluation batch_size
                             gradient_accumulation_steps=args.gradient_accumulation_steps,  # gradient accumulation steops
                             learning_rate=args.learning_rate,  # learning rate
                             warmup_steps=args.warmup_steps,  # warmup steps
                             num_train_epochs=args.num_train_epochs,  # finetune epochs
                             save_strategy="steps",  # save step
                             evaluation_strategy="steps",  # evaluation strategy
                             load_best_model_at_end=True,
                             fp16=args.fp16,  # use fp16 for training?
                             report_to=["tensorboard"],  # tensorboard log
                             save_steps=args.save_steps,  # save steps
                             eval_steps=args.eval_steps,  # evaluation steps
                             torch_compile=args.use_compile, # use Pytorch2.0 compiler
                             save_total_limit=5,  # save latest for resume
                             optim='adamw_torch',  # optimizer
                             ddp_find_unused_parameters=False if ddp else None,  # distributed training
                             dataloader_num_workers=args.num_workers,  # number of works
                             logging_steps=args.logging_steps,  # log steps
                             remove_unused_columns=False,  # remove unused columns
                             label_names=["labels"]) 

if training_args.local_rank == 0 or training_args.local_rank == -1:
    print('=' * 90)
    model.print_trainable_parameters()
    print('=' * 90)

# define trainer
trainer = Seq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=test_dataset,
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         callbacks=[SavePeftModelCallback])
model.config.use_cache = False
trainer._load_from_checkpoint = load_from_checkpoint

# start training ...
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# save the last model result
trainer.save_state()
if training_args.local_rank == 0 or training_args.local_rank == -1:
    model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
