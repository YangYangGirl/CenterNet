var=lm_shuffle16y10
var_arch=shuffle16y10
cd ../src
# train
python main.py --metric loss --down_ratio 16 --task landmark --arch $var_arch --exp_id $var --dataset landmark --num_epochs 1 --batch_size 64 --master_batch 64 --lr 5e-4  --gpus 0 --num_workers 1
# test
python test.py --down_ratio 16 --batch_size 1 --arch $var_arch --task landmark --exp_id $var --dataset landmark --keep_res --resume
# flip test
#python test.py --down_ratio 4 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume --flip_test
cd ..
