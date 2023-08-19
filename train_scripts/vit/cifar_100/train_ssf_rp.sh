
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=0,1,2,3  python  -u -m torch.distributed.launch --nproc_per_node=4  --master_port=12345  \
	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/ssf_rp_drop \
	--amp --tuning-mode ssf --pretrained --no-save --seed -1\
    --use-relax-param --reparam-interval 50  --reparam-init-method constant\
    --reparam-way drop --drop-rate 0.1 \

# source DATA_PATH.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -u -m torch.distributed.launch --nproc_per_node=4  --master_port=12346  \
# 	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/ssf_rp_drop \
# 	--amp --tuning-mode ssf --pretrained --no-save --seed -1\
#     --use-relax-param --reparam-interval 50  --reparam-init-method constant\
#     --reparam-way drop --drop-rate 0.2 \

# source DATA_PATH.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3  python  -u -m torch.distributed.launch --nproc_per_node=4  --master_port=12347  \
# 	train.py ${CIFAR100_PATH}/ --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 100 \
# 	--opt adamw  --weight-decay 0.05 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--model-ema --model-ema-decay 0.99992  \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/cifar_100/ssf_rp_drop \
# 	--amp --tuning-mode ssf --pretrained --no-save --seed -1\
#     --use-relax-param --reparam-interval 50  --reparam-init-method constant\
#     --reparam-way drop --drop-rate 0.3 \