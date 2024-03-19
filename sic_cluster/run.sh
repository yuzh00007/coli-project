nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python

#python utils/parse_trees.py

# for testing
#python main.py --sample_size 100 --epoch 2 --per_device_train_batch_size 16 --learning_rate 5e-5

# actual runs
# run a first baseline, then train and evaluate
python main.py --baseline True --epoch 10 --per_device_train_batch_size 8 --learning_rate 5e-5
python main.py --epoch 10 --per_device_train_batch_size 8 --learning_rate 5e-5

# run with larger batch sizes
python main.py --baseline True --epoch 10 --per_device_train_batch_size 32 --learning_rate 5e-5
python main.py --epoch 10 --per_device_train_batch_size 32 --learning_rate 5e-5

python main.py --baseline True --epoch 10 --per_device_train_batch_size 32 --learning_rate 5e-5
python main.py --epoch 10 --per_device_train_batch_size 32 --learning_rate 5e-5
