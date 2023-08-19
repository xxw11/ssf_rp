
source DATA_PATH.sh
CUDA_VISIBLE_DEVICES=$1  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=$2  \
	train_regular.py ${VTAB_PATH}/dsprites_ori  --dataset dsprites_ori --num-classes 16  --no-aug   --direct-resize   --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  ${OUTPUT_PATH}/vit_base_patch16_224_in21k/vtab/dsprites_ori/pruning \
	--amp --tuning-mode ssf --pretrained --no-save --seed 1 \
	--reg 1e-4 \
	--model-ema --model-ema-decay 0.9	