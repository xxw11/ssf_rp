
# source DATA_PATH.sh
#   	python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12222  \
# 	train.py ${VTAB_PATH}/caltech101  --dataset caltech101 --num-classes 102  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
#     --batch-size 32 --epochs 500 \
# 	--opt adamw  --weight-decay 5e-2 \
#     --warmup-lr 1e-7 --warmup-epochs 10  \
#     --lr 1e-3 --min-lr 1e-8 \
#     --drop-path 0.1 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
# 	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/caltech101/ssf \
# 	--amp  --tuning-mode ssf --pretrained  \
# 	--reparam-way tanh 





#8.5
source DATA_PATH.sh
  	python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12222  \
	train.py ${VTAB_PATH}/caltech101  --dataset caltech101 --num-classes 102  --no-aug  --direct-resize  --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 500 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/caltech101/ssf \
	--amp  --tuning-mode ssf --pretrained  \
	--reparam-way tanh --use-relax-param