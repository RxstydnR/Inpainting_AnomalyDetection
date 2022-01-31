
train_data="train_dir"
test_data="test_dir"
savepath="result_demo"
img_size=256

python train.py \
    --dataset ${train_data} \
    --SAVEPATH ${savepath} \
    --img_size ${img_size} \
    --batchSize 8 \
    --nEpochs 300 \
    --lr 0.0001 \
    --GPUs 1 2 3


python test.py \
    --dataset ${test_data}   \
    --SAVEPATH ${savepath}  \
    --img_size ${img_size}   \
    --model_weight ${savepath}/model_weights.h5 \
    --GPUs 1