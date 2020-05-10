var=lm_shuffle16y10
var_task=landmark
var_arch=shuffle16y10
cd ../src
# train
python main.py --metric loss --down_ratio 16 --task landmark --arch $var_arch --exp_id $var --dataset landmark --num_epochs 200 --batch_size 64 --master_batch 64 --lr 5e-4  --gpus 0 --num_workers 1

python test_widerface.py --down_ratio 16 --batch_size 1 --arch $var_arch   --num_class 1 --dataset landmark --task landmark --load_model "../exp/"$var_task"/"$var"/model_best.pth"  --outdir  "../output/"$var"/"
cd widerface_evaluate
# multi scale test
python3 evaluation.py -p "../../output/"${var}"/" -g ./ground_truth
cd ..


# test
#python test.py --down_ratio 16 --batch_size 1 --arch $var_arch --task landmark --exp_id $var --dataset landmark --keep_res --resume
# flip test
#python test.py --down_ratio 4 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume --flip_test
#cd ..
