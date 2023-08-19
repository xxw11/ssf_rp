name="train_ssf"
sleep $3
bash train_scripts/vit/vtab/caltech101/${name}.sh $1 $2
bash train_scripts/vit/vtab/cifar_100/${name}.sh $1 $2
bash train_scripts/vit/vtab/clevr_count/${name}.sh $1 $2
bash train_scripts/vit/vtab/clevr_dist/${name}.sh $1 $2
bash train_scripts/vit/vtab/diabetic_retinopathy/${name}.sh $1 $2
bash train_scripts/vit/vtab/dmlab/${name}.sh $1 $2
bash train_scripts/vit/vtab/dsprites_loc/${name}.sh $1 $2
bash train_scripts/vit/vtab/dsprites_ori/${name}.sh $1 $2
bash train_scripts/vit/vtab/dtd${name}.sh $1 $2
bash train_scripts/vit/vtab/eurosat/${name}.sh $1 $2
bash train_scripts/vit/vtab/flowers102${name}.sh $1 $2
bash train_scripts/vit/vtab/kitti/${name}.sh $1 $2
bash train_scripts/vit/vtab/patch_camelyon/${name}.sh $1 $2
bash train_scripts/vit/vtab/pets/${name}.sh $1 $2
bash train_scripts/vit/vtab/resisc45/${name}.sh $1 $2
bash train_scripts/vit/vtab/smallnorb_azi/${name}.sh $1 $2
bash train_scripts/vit/vtab/smallnorb_ele/${name}.sh $1 $2
bash train_scripts/vit/vtab/sun397/${name}.sh $1 $2
bash train_scripts/vit/vtab/svhn/${name}.sh $1 $2