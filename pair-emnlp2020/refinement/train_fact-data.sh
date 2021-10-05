python train.py \
    --n-gpus=1 \
    --domain=fact \
    --setup=pair-full \
    --train-set=train \
    --valid-set=valid \
    --eval-batch-size=1 \
    --train-batch-size=1 \
    --num-train-epochs=20 \
    --ckpt-dir=../checkpoints/fact-data/full/ \
    --tensorboard-dir=fact-data-full \
    --quiet \
    --fp16