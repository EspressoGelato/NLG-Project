CUDA_VISIBLE_DEVICES=0 python train.py \
        --data-path=../data/fact_data/ \
        --domain=fact \
        --exp-name=full \
        --save-interval=5 \
        --max-epoch=40 \
        --warmup-updates=5000 \
        --train-set=train \
        --valid-set=valid \
        --tensorboard-logdir=tboard/full \
        --lr=5e-4 \
        --max-samples=16 \
        --predict-keyphrase-offset \
	--quiet
