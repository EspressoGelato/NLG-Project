 python train.py \
        --data-path=../data/data_release/ \
        --domain=arggen \
        --exp-name=full \
        --save-interval=5 \
        --max-epoch=40 \
        --warmup-updates=5000 \
        --train-set=train \
        --valid-set=dev \
        --tensorboard-logdir=tboard/full \
        --lr=5e-4 \
        --quiet \
        --max-samples=16 \
        --predict-keyphrase-offset \
