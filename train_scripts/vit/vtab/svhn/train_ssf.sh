
source DATA_PATH.sh
  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=11122  \
	train.py ${VTAB_PATH}/svhn  --dataset svhn --num-classes 10  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/svhn/ssf \
	--amp --tuning-mode ssf --pretrained  \
	--reparam-way tanh --use-relax-param