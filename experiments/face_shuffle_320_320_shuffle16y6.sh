var=shuffle_320_320_shuffle16y6
var_arch=shuffle16y6
cd ../src
# train
python main.py --task ctdet --dataset face  --num_epochs 120 --batch_size  128 --master_batch 128   --lr 1.25e-4 --gpus 1  --metric loss  --exp_id  $var  --arch $var_arch
# produce output
python test_widerface.py --arch $var_arch   --num_class 1 --dataset face --task ctdet --load_model "../exp/ctdet/"$var"/model_best.pth"  --outdir "../output/"$var"/"
# evaluate output
cd widerface_evaluate
# multi scale test
python3 evaluation.py -p "../../output/"$var"/" -g ./ground_truth
cd ..
