var=Mb_Tiny_RFB_d16_320
cd ../src
# train
python main.py --down_ratio 16 --task ctdet --dataset face  --num_epochs 300 --batch_size  128  --master_batch 128   --lr 1.25e-4 --gpus 1  --metric loss  --exp_id  $var  --arch MbTinyRFB_5
# produce output
python test_widerface.py --down_ratio 16 --batch_size 1 --arch MbTinyRFB_5   --num_class 1 --dataset face --task ctdet --load_model "../exp/ctdet/"$var"/model_best.pth"  --outdir  "../output/"$var"/"
# evaluate output
cd widerface_evaluate
# multi scale test
python3 evaluation.py -p "../../output/"${var}"/" -g ./ground_truth
cd ..
