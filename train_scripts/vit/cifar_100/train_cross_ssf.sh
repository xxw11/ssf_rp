#7.29 21:21
source DATA_PATH.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# 	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/cross_ssf \
# 	--amp --tuning-mode cross_ssf --pretrained --no-save --seed -1  \

#7.30 11:20
# Ablation Study : SSF ONLY
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# 	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/cross_ssf \
# 	--amp --tuning-mode ssf --pretrained --no-save --seed -1  \

#7.30 16:10
# Ablation Study : flash attention fp16 bf16 only
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# 	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 2e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/cross_ssf \
# 	--amp --tuning-mode ssf --pretrained --no-save --seed -1  \

#7.30 19:49
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# 	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/cross_ssf \
# 	--amp --tuning-mode cross_ssf --pretrained --no-save --seed -1  \

#7.31 10:07 learning rate / 2
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# 	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 5e-4 --min-lr 5e-9 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/cross_ssf \
# 	--amp --tuning-mode cross_ssf --pretrained --no-save --seed -1  \

#7.31 13:24 learning rate / 5
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
# 	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 2e-4 --min-lr 2e-9 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/cross_ssf \
# 	--amp --tuning-mode cross_ssf --pretrained --no-save --seed -1  \