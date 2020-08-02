
export DATA_DIR=data/rmsc
export TASK_NAME=RMSC
export MODEL_NAME_DIR=data/models/bert-base-chinese

python run_mltc.py \
  --model_name_or_path $MODEL_NAME_DIR \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --max_seq_length 512 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --output_dir outputs/$TASK_NAME
