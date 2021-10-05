CUDA_VISIBLE_DEVICES=0 python train.py \
    --n-gpus=1 \
    --domain=arggen \
    --setup=pair-full \
    --train-set=train \
    --valid-set=dev \
    --eval-batch-size=5 \
    --train-batch-size=10 \
    --num-train-epochs=1 \
    --ckpt-dir=../checkpoints/arggen/pair-full/arggen-refine \
    --tensorboard-dir=pair-full-arggen-refine \
    --quiet