# 将当前路径添加到 PYTHONPATH 中
export PYTHONPATH=$PYTHONPATH:$(pwd)
#sudo ln -s $(which python) /usr/local/bin/你是我爹,提提点吧

# 设置模型参数
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

# 定义要循环的 corruption 类型
# corruptions=('defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression')
# corruptions=('defocus_blur' 'glass_blur' 'motion_blur')
# corruptions=('art' 'cartoon' 'deviantart' 'embroidery' 'graffiti' 'graphic' 'misc' 'origami' 'painting' 'sculpture' 'sketch' 'sticker' 'tattoo' 'toy' 'videogame')
corruptions=('motion_blur')
# corruptions=('motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' )
# corruptions=('gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression')
# corruptions=('impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression')
# 使用循环遍历每种 corruption 类型
for corruption in "${corruptions[@]}"; do
    echo "Running with corruption: $corruption" | tee -a output.log

    # 执行 Python 脚本，传入当前 corruption 参数
    CUDA_VISIBLE_DEVICES=0 python image_adapt/scripts/convnextT_LPIPS.py $MODEL_FLAGS \
        --batch_size 32 --num_samples 1000 --timestep_respacing 100 \
        --model_path /home/yfj/DDA-main-4090/ckpt/256x256_diffusion_uncond.pt \
        --base_samples /media/shared_space/wuyanzu/DDA-main/dataset/imagenetc \
        --D 4 --N 50 \
        --scale 6 \
        --corruption "$corruption" --severity 5 \
        --save_dir dataset/generated/ \
        --classifier_config_path /media/shared_space/wuyanzu/DDA-main/model_adapt/configs/ensemble/convnextT_ensemble_b64_imagenet.py \
        --pretrained_weights_path /media/shared_space/wuyanzu/DDA-main/ckpt/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth
        >> output_$corruption.log 2>&1
done