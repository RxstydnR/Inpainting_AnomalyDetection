# Need to check the availability of the GPUs to be used.
# -> nvidia-smi

# 並列処理 (並列処理で学習を行う場合, GPUの数を増やすほど学習の開始までが非常に遅いことに注意.)
# おそらくTFとCUDAのverを上げる必要がある

train_data="/data/Users/katafuchi/DENSO/512_512_normal/normal_512_512_input/FP25"
test_data="/data/Users/katafuchi/DENSO/512_512_normal/normal_512_512_input/FP25"
savepath="result_demo"
img_size=256
mask_size=32

python train.py \
    --dataset ${train_data} \
    --SAVEPATH ${savepath} \
    --img_size ${img_size} \
    --mask_size ${mask_size} \
    --batchSize 8 \
    --nEpochs 300 \
    --lr 0.0001 \
    --GPUs 5 6 7

# 現状test.pyは, 画像の保存などの関数でbatchsize=1のみに対応した形式となっている.
# そのため, Testでは並列化の恩恵は得られてない可能性あり.

# python test.py \
#     --dataset ${test_data}   \
#     --SAVEPATH ${savepath}  \
#     --img_size ${img_size}   \
#     --mask_size ${mask_size} \
#     --model_weight ${savepath}/model_weights.h5 \
#     --GPUs 1