# bash /data7/binbinyang/FSDA-DETR/scripts/xView2DOTA/DINO_train.sh
export CUDA_VISIBLE_DEVICES=0 && python main.py \
	--output_dir logs/DINO_xView2DOTA_FSDA/R50-MS4 -c /data7/binbinyang/FSDA-DETR/config/FSDA/xView2DOTA/DINO_4scale.py \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

