var=lm_shuffle16y6
var_arch=shuffle16y6
cd ../src
# train
python main.py --metric loss --down_ratio 16 --task landmark --arch $var_arch --exp_id $var --dataset landmark --num_epochs 300 --batch_size 64 --master_batch 1 --lr 5e-4  --gpus 0 --num_workers 16
# test
#python test.py --down_ratio 4 --batch_size 1 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume
# flip test
#python test.py --down_ratio 4 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume --flip_test
cd ..
