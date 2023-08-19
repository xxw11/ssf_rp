#20230804-193549-vit_base_patch16_224_in21k-224
source DATA_PATH.sh
#  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=14444  \
# 	train.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 500 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/ssf \
# 	--amp  --tuning-mode ssf --pretrained  \
# 	--reparam-way tanh


#20230805-101610-vit_base_patch16_224_in21k-224
# source DATA_PATH.sh
#  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=14444  \
# 	train.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 500 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/ssf \
# 	--amp  --tuning-mode ssf --pretrained  \
# 	--reparam-way tanh --use-relax-param


#0805 1016
#0805 1410 取消了前面corss_ssf的重参数化
tuning_mode="cross_ssf"
python  -m torch.distributed.launch --nproc_per_node=4  --master_port=14444  \
	train.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 500 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/${tuning_mode} \
	--amp  --tuning-mode ${tuning_mode} --pretrained  \
	--reparam-way tanh --use-relax-param