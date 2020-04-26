cd ../src
# train
#python main.py --down_ratio 4 --task landmark  --exp_id dla_1x --dataset landmark --batch_size 1 --master_batch 1 --lr 5e-4  --gpus 0 --num_workers 16
# test
python test.py --down_ratio 4 --batch_size 1 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume
# flip test
python test.py --down_ratio 4 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume --flip_test
cd ..
