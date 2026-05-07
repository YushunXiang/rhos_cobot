#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate lerobot

# 遍历符合条件的文件夹
for log_dir in examples/piper_real/logs/0428*/; do
    echo "Processing directory: $log_dir"
    
    # 检查 images 目录是否存在
    img_dir="${log_dir}model_input_observation/images"
    if [ ! -d "$img_dir" ]; then
        echo "Images directory $img_dir not found, skipping..."
        continue
    fi
    
    # 进入 image 目录，生成各视角的MP4文件
    pushd "$img_dir" > /dev/null
    
    echo "  Generating high view video..."
    ffmpeg -framerate 25 -start_number 1 -i "cam_high_%d.png" \
           -c:v libx264 -pix_fmt yuv420p -r 25 -y "output_high.mp4" -loglevel error
           
    echo "  Generating left wrist video..."
    ffmpeg -framerate 25 -start_number 1 -i "cam_left_wrist_%d.png" \
           -c:v libx264 -pix_fmt yuv420p -r 25 -y "output_left_wrist.mp4" -loglevel error
           
    echo "  Generating right wrist video..."
    ffmpeg -framerate 25 -start_number 1 -i "cam_right_wrist_%d.png" \
           -c:v libx264 -pix_fmt yuv420p -r 25 -y "output_right_wrist.mp4" -loglevel error
           
    echo "  Combining videos..."
    ffmpeg -i "output_left_wrist.mp4" -i "output_high.mp4" -i "output_right_wrist.mp4" \
           -filter_complex "[0:v][1:v][2:v]hstack=inputs=3" -c:v libx264 -pix_fmt yuv420p -y "../combined_output.mp4" -loglevel error
           
    popd > /dev/null
    echo "Finished $log_dir"
    echo "----------------------------------------"
done
echo "All done!"
