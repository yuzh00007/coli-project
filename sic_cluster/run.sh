nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python

#python utils/parse_trees.py
python main.py --sample_size 100 --baseline False --epoch 2 --per_device_train_batch_size 16 --learning_rate 5e-5
