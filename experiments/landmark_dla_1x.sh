var=lm_dla_1x

cd ../src
# train
python main.py --load_model ../models/ctdet_coco_dla_2x.pth --metric loss --down_ratio 4 --task landmark  --exp_id $var --dataset landmark --batch_size 16 --master_batch 16 --lr 5e-4  --gpus 0 --num_workers 16
# test
python test.py --down_ratio 4 --batch_size 1 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume
# flip test
python test.py --down_ratio 4 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume --flip_test
cd ..
