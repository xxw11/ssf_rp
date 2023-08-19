
source DATA_PATH.sh
# CUDA_VISIBLE_DEVICES=$1 python  -m torch.distributed.launch --nproc_per_node=2  --master_port=$2  \
# 	train.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/ssf \
# 	--amp  --tuning-mode ssf --pretrained  \

#8.4 10
# 0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# train.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 500 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/cross_ssf \
# 	--amp --tuning-mode cross_ssf --pretrained --no-save --seed -1  \

#8.4 12 learning rate 5e-4 for small dataset 
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# train.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 500 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-4 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/cross_ssf \
# 	--amp --tuning-mode cross_ssf --pretrained --no-save --seed -1  \


#8.4 12 learning rate 5e-4 for small dataset 
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# train.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 500 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-4 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/cross_ssf \
# 	--amp --tuning-mode cross_ssf --pretrained --no-save --seed 42  \


#8.4 15 1.3*Learning Rate
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# train.py ${VTAB_PATH}/cifar/  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 500 \
# 	--opt adamw  --weight-decay 5e-5 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 8e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/cifar_100/cross_ssf \
# 	--amp --tuning-mode cross_ssf --pretrained --no-save --seed -1  \

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