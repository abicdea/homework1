import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from swift.llm import (
DatasetName, InferArguments, ModelType, SftArguments,
infer_main, sft_main, app_ui_main, merge_lora_main
)
model_type = ModelType.qwen_1_8b
sft_args = SftArguments(
model_type=model_type,
sft_type='lora',
train_dataset_sample=2000,
dataset=[DatasetName.jd_sentiment_zh],
output_dir='output')
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()
infer_args = InferArguments(
ckpt_dir=best_model_checkpoint,
load_dataset_config=True,
do_sample=False)
result = infer_main(infer_args)
