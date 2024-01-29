---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- sentiment-classification
---
# <情感分类模型>介绍

本项目使用 SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) 微调框架，选择采用LoRA微调方法，对 通义千问1.8B 基础模型进行微调，以实现对句子的情感分类任务。


## 实验环境

modelscope平台免费实例-GPU环境（8核 32GB 显存24G (A10显卡)；预装 ModelScope Library; 预装镜像 Ubuntu22.04 cuda12.1.0 python3.10 pytorch2.1.2 tensorflow2.14.0)

需手动安装微调框架库 swift 以及所需的其他第三方库，可以通过下方代码进行环境准备：

    git clone https://github.com/modelscope/swift.git 
    cd swift
    pip install -e .[llm]

## 训练方法

### 数据集

京东商品评论情感分析数据集 (jd_sentiment_zh)

### 微调的基础模型

通义千问1.8B (qwen_1_8b) 

### 微调方法

LoRA

### 其他超参数

```json
sft参数 (sfc_args.json)
{
  "model_type": "qwen-1_8b",
  "model_id_or_path": "qwen/Qwen-1_8B",
  "model_revision": "master",
  "model_cache_dir": null,
  "sft_type": "lora",
  "freeze_parameters": 0.0,
  "additional_trainable_parameters": [],
  "tuner_backend": "swift",
  "template_type": "default-generation",
  "output_dir": "/mnt/workspace/output/qwen-1_8b/v0-20240128-024328",
  "add_output_dir_suffix": true,
  "ddp_backend": "nccl",
  "seed": 42,
  "resume_from_checkpoint": null,
  "dtype": "bf16",
  "dataset": [
    "jd-sentiment-zh"
  ],
  "dataset_seed": 42,
  "dataset_test_ratio": 0.01,
  "train_dataset_sample": 2000,
  "val_dataset_sample": null,
  "system": null,
  "max_length": 2048,
  "truncation_strategy": "delete",
  "check_dataset_strategy": "none",
  "custom_train_dataset_path": [],
  "custom_val_dataset_path": [],
  "self_cognition_sample": 0,
  "model_name": [
    null,
    null
  ],
  "model_author": [
    null,
    null
  ],
  "quantization_bit": 0,
  "bnb_4bit_comp_dtype": "bf16",
  "bnb_4bit_quant_type": "nf4",
  "bnb_4bit_use_double_quant": true,
  "lora_target_modules": [
    "c_attn"
  ],
  "lora_rank": 8,
  "lora_alpha": 32,
  "lora_dropout_p": 0.05,
  "lora_bias_trainable": "none",
  "lora_modules_to_save": [],
  "lora_dtype": "fp32",
  "neftune_noise_alpha": null,
  "gradient_checkpointing": true,
  "deepspeed_config_path": null,
  "batch_size": 1,
  "eval_batch_size": 1,
  "num_train_epochs": 1,
  "max_steps": -1,
  "optim": "adamw_torch",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "gradient_accumulation_steps": 16,
  "max_grad_norm": 0.5,
  "predict_with_generate": false,
  "lr_scheduler_type": "linear",
  "warmup_ratio": 0.05,
  "eval_steps": 50,
  "save_steps": 50,
  "save_only_model": false,
  "save_total_limit": 2,
  "logging_steps": 5,
  "dataloader_num_workers": 1,
  "push_to_hub": false,
  "hub_model_id": "qwen-1_8b-lora",
  "hub_private_repo": true,
  "push_hub_strategy": "push_best",
  "hub_token": null,
  "test_oom_error": false,
  "disable_tqdm": false,
  "lazy_tokenize": false,
  "preprocess_num_proc": 1,
  "use_flash_attn": null,
  "ignore_args_error": false,
  "check_model_is_latest": true,
  "logging_dir": "/mnt/workspace/output/qwen-1_8b/v0-20240128-024328/runs",
  "report_to": [
    "all"
  ],
  "acc_strategy": "token",
  "save_on_each_node": true,
  "evaluation_strategy": "steps",
  "save_strategy": "steps",
  "save_safetensors": true,
  "max_new_tokens": 2048,
  "do_sample": true,
  "temperature": 0.3,
  "top_k": 20,
  "top_p": 0.7,
  "repetition_penalty": 1.0,
  "num_beams": 1,
  "per_device_train_batch_size": null,
  "per_device_eval_batch_size": null,
  "only_save_model": null,
  "neftune_alpha": null,
  "torch_dtype": "torch.bfloat16",
  "fp16": false,
  "bf16": true,
  "bnb_4bit_compute_dtype": "torch.bfloat16",
  "load_in_4bit": false,
  "load_in_8bit": false,
  "train_sampler_random": true,
  "deepspeed": null
}
```

```json
训练参数 (training_args.json)
{
  "output_dir": "/mnt/workspace/output/qwen-1_8b/v0-20240128-024328",
  "overwrite_output_dir": false,
  "do_train": false,
  "do_eval": true,
  "do_predict": false,
  "evaluation_strategy": "steps",
  "prediction_loss_only": false,
  "per_device_train_batch_size": 1,
  "per_device_eval_batch_size": 1,
  "per_gpu_train_batch_size": null,
  "per_gpu_eval_batch_size": null,
  "gradient_accumulation_steps": 16,
  "eval_accumulation_steps": null,
  "eval_delay": 0,
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-08,
  "max_grad_norm": 0.5,
  "num_train_epochs": 1,
  "max_steps": -1,
  "lr_scheduler_type": "linear",
  "lr_scheduler_kwargs": {},
  "warmup_ratio": 0.05,
  "warmup_steps": 0,
  "log_level": "passive",
  "log_level_replica": "warning",
  "log_on_each_node": true,
  "logging_dir": "/mnt/workspace/output/qwen-1_8b/v0-20240128-024328/runs",
  "logging_strategy": "steps",
  "logging_first_step": true,
  "logging_steps": 5,
  "logging_nan_inf_filter": true,
  "save_strategy": "steps",
  "save_steps": 50,
  "save_total_limit": 2,
  "save_safetensors": true,
  "save_on_each_node": true,
  "save_only_model": false,
  "no_cuda": false,
  "use_cpu": false,
  "use_mps_device": false,
  "seed": 42,
  "data_seed": null,
  "jit_mode_eval": false,
  "use_ipex": false,
  "bf16": true,
  "fp16": false,
  "fp16_opt_level": "O1",
  "half_precision_backend": "auto",
  "bf16_full_eval": false,
  "fp16_full_eval": false,
  "tf32": null,
  "local_rank": 0,
  "ddp_backend": "nccl",
  "tpu_num_cores": null,
  "tpu_metrics_debug": false,
  "debug": [],
  "dataloader_drop_last": false,
  "eval_steps": 50,
  "dataloader_num_workers": 1,
  "past_index": -1,
  "run_name": "/mnt/workspace/output/qwen-1_8b/v0-20240128-024328",
  "disable_tqdm": false,
  "remove_unused_columns": false,
  "label_names": null,
  "load_best_model_at_end": true,
  "metric_for_best_model": "loss",
  "greater_is_better": false,
  "ignore_data_skip": false,
  "fsdp": [],
  "fsdp_min_num_params": 0,
  "fsdp_config": {
    "min_num_params": 0,
    "xla": false,
    "xla_fsdp_grad_ckpt": false
  },
  "fsdp_transformer_layer_cls_to_wrap": null,
  "deepspeed": null,
  "label_smoothing_factor": 0.0,
  "optim": "adamw_torch",
  "optim_args": null,
  "adafactor": false,
  "group_by_length": false,
  "length_column_name": "length",
  "report_to": [
    "tensorboard"
  ],
  "ddp_find_unused_parameters": null,
  "ddp_bucket_cap_mb": null,
  "ddp_broadcast_buffers": null,
  "dataloader_pin_memory": true,
  "dataloader_persistent_workers": false,
  "skip_memory_metrics": true,
  "use_legacy_prediction_loop": false,
  "push_to_hub": false,
  "resume_from_checkpoint": null,
  "hub_model_id": "qwen-1_8b-lora",
  "hub_strategy": "every_save",
  "hub_token": null,
  "hub_private_repo": true,
  "hub_always_push": false,
  "gradient_checkpointing": true,
  "gradient_checkpointing_kwargs": null,
  "include_inputs_for_metrics": false,
  "fp16_backend": "auto",
  "push_to_hub_model_id": null,
  "push_to_hub_organization": null,
  "push_to_hub_token": null,
  "mp_parameters": "",
  "auto_find_batch_size": false,
  "full_determinism": false,
  "torchdynamo": null,
  "ray_scope": "last",
  "ddp_timeout": 1800,
  "torch_compile": false,
  "torch_compile_backend": null,
  "torch_compile_mode": null,
  "dispatch_batches": null,
  "split_batches": false,
  "include_tokens_per_second": false,
  "include_num_input_tokens_seen": false,
  "neftune_noise_alpha": null,
  "sortish_sampler": true,
  "predict_with_generate": false,
  "generation_max_length": null,
  "generation_num_beams": null,
  "generation_config": "GenerationConfig {\n  \"chat_format\": \"raw\",\n  \"do_sample\": true,\n  \"eos_token_id\": 151643,\n  \"max_new_tokens\": 2048,\n  \"pad_token_id\": 151643,\n  \"stop_words_ids\": [\n    [\n      151643\n    ]\n  ],\n  \"temperature\": 0.3,\n  \"top_k\": 20,\n  \"top_p\": 0.7\n}\n",
  "train_sampler_random": true,
  "push_hub_strategy": "push_best",
  "acc_strategy": "token",
  "additional_saved_files": [],
  "distributed_state": "Distributed environment: NO\nNum processes: 1\nProcess index: 0\nLocal process index: 0\nDevice: cuda\n",
  "_n_gpu": 1,
  "__cached__setup_devices": "device(type='cuda', index=0)",
  "deepspeed_plugin": null
}
```

## 示例代码

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from swift.llm import (DatasetName, InferArguments, ModelType, SftArguments,infer_main, sft_main, app_ui_main, merge_lora_main)
```

```python
# sft

model_type = ModelType.qwen_1_8b

sft_args = SftArguments(model_type=model_type,sft_type='lora',train_dataset_sample=2000,dataset=[DatasetName.jd_sentiment_zh],output_dir='output')

result = sft_main(sft_args)

best_model_checkpoint = result['best_model_checkpoint']

print(f'best_model_checkpoint: {best_model_checkpoint}')

torch.cuda.empty_cache()
```

```python
# inference

inf_args = InferArguments(ckpt_dir=best_model_checkpoint,load_dataset_config=True,do_sample=False,max_new_tokens=5)

result = infer_main(inf_args)

torch.cuda.empty_cache()
```
