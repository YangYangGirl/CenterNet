var=shuffle_10_320_320_dw_5_5_d_16_6
cd ../src
# train
#python main.py --task ctdet --dataset face  --num_epochs 300 --batch_size  32 --master_batch 32   --lr 1.25e-4 --gpus 1  --metric loss  --exp_id  $var  --arch  shuffleDw5_5
# produce output
CUDA_VISIBLE_DEVICES=0 python test_widerface.py --batch_size 1 --arch shuffleDw5_5   --num_class 1 --dataset face --task ctdet --load_model "../exp/ctdet/"$var"/model_best.pth"  --outdir  "../output/"$var"/"
# evaluate output
cd widerface_evaluate
# multi scale test
python3 evaluation.py -p "../../output/"${var}"/" -g ./ground_truth
cd ..
