nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python

#python utils/parse_trees.py
python main.py --sample_size 50 --baseline False --epoch 5 --per_device_train_batch_size 16
