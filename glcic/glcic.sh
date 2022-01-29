
train_data="train_data" 
test_data="test_data" 
savepath="result"
img_size=256
mask_size=32

python train.py \
    --dataset ${train_data} \
    --SAVEPATH ${savepath} \
    --img_size ${img_size} \
    --mask_size ${mask_size} \
    --batchSize 8 \
    --nEpochs 10 \
    --lr 0.0001 \
    --GPUs 1 2 

python test.py \
    --dataset ${test_data} \
    --SAVEPATH ${savepath}\
    --img_size ${img_size} \
    --mask_size ${mask_size} \
    --model_weight ${savepath}/generator_weights.h5 \
    --GPUs 1 2
