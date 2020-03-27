var=shuffle_5_128_128_d_16_6_dw_3_3_lr_low
cd ../src
# train
#python main.py --task ctdet --dataset face  --num_epochs 300 --batch_size  32 --master_batch 32   --lr 1.25e-5 --gpus 1  --metric loss  --exp_id  $var  --arch  shuffle_5
# produce output
python test_widerface.py --arch shuffle_5 --num_class 1 --dataset face --task ctdet --load_model "../exp/ctdet/"$var"/model_best.pth"  --outdir "../output/"$var"/"
# evaluate output
cd widerface_evaluate
# multi scale test
python3 evaluation.py -p "../../output/"$var"/" -g ./ground_truth
cd ../../
