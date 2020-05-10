var=lm_dla_1x
var_arch=dla_34
var_task=landmark
cd ../src
# train
#python main.py --metric loss --down_ratio 4 --task landmark --num_epochs 150 --arch $var_arch --exp_id $var --dataset landmark --load_model ../models/ctdet_coco_dla_2x.pth --batch_size 6 --master_batch 6 --lr 5e-5  --gpus 0 --num_workers 1

#python test_widerface.py --down_ratio 16 --batch_size 1 --arch $var_arch   --num_class 1 --dataset landmark --task landmark --load_model "../exp/"$var_task"/"$var"/model_best.pth"  --outdir  "../output/"$var"/"
#cd widerface_evaluate
# multi scale test
#python3 evaluation.py -p "../../output/"${var}"/" -g ./ground_truth
#cd ..


# test
#python test.py --down_ratio 4 --batch_size 1 --arch $var_arch --task landmark --exp_id $var --dataset landmark --keep_res --resume
# flip test
#python test.py --down_ratio 4 --task landmark --exp_id dla_1x --dataset landmark --keep_res --resume --flip_test
#cd ..
