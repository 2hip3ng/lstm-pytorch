
# CUDA_VISIBLE_DEVICES=0 \
python ./../run_classifier.py  \
    --data_dir /home/bupt/HaHa/wzp/code/lstm-pytorch/sohu_text_classification/cnews_data/clean_data \
    --task_name cnews \
    --vocab_path /home/bupt/HaHa/wzp/code/lstm-pytorch/sohu_text_classification/cnews_data/vocab/vocab.txt \
    --output_dir /home/bupt/HaHa/wzp/code/lstm-pytorch/sohu_text_classification/output \
    --max_seq_length 300 \
    --do_train \
    --do_eval \
	--num_train_epochs 1 \
	--bidirectional true \
    # --no_cuda \



