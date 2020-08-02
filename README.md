# Transformers-MLTC

This repo contains several Transformers for Multi-Label Text Classification (MLTC). We follow the [huggingface's implementation](https://github.com/huggingface/transformers).

## Requirements
This code is tested on Python 3.6+, PyTorch 1.5+ and transformers 3.0.2+.
```
pip install -r requirements.txt
```

## Data & Models
You can download the `AAPD` and `RMSC-V2` datasets from the [link](https://mega.nz/file/dLgFTAjB#vgfRg3IcaB17I4iKfgU5aYORabogc5mc2-QiYFvFLs8) and decompress to the `data` directory. Also, custom processors in `datasets` are also effective.

Pre-trained models can be download from [huggingface's model zoos](https://huggingface.co/models).
- [bert-base-uncased](https://huggingface.co/bert-base-uncased#list-files)
- [bert-base-chinese](https://huggingface.co/bert-base-chinese#list-files)

## Details
The bert-based model use the hidden state of the first token (e.g. [CLS]) in last layer, and several logistic regression (e.g. linear + sigmoid) to predict the multiple labels. a binary cross entropy loss is used.

## Runs

We provide two tasks: `AAPD` and `RMSC` for MLTC task.

```
export DATA_DIR=data/aapd
export MODEL_NAME_DIR=data/models/bert-base-uncased
export TASK_NAME=AAPD

python run_mltc.py \
  --model_name_or_path $MODEL_NAME_DIR \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --output_dir outputs/$TASK_NAME/
```

## Acknowledgement
Our implementation is based on the [huggingface's transformers](https://github.com/huggingface/transformers).
