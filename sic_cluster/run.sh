nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python

#python utils/parse_trees.py

# for testing
#python main.py --sample_size 100 --epoch 3 --per_device_train_batch_size 8 --learning_rate 5e-5

# basic run
# run a first baseline, then train and evaluate
python main.py --baseline True
python main.py --epoch 10 --per_device_train_batch_size 8 --learning_rate 5e-5

# run with smaller lr
#python main.py --epoch 10 --per_device_train_batch_size 8 --learning_rate 1e-5

# run with larger batch
#python main.py --epoch 10 --per_device_train_batch_size 16 --learning_rate 5e-5

# run with fewer epochs
#python main.py --epoch 3 --per_device_train_batch_size 8 --learning_rate 5e-5

# run with more epochs
#python main.py --epoch 30 --per_device_train_batch_size 8 --learning_rate 5e-5

