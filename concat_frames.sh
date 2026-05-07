#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate lerobot

if [ "$#" -ne 2 ]; then
    echo "用法: $0 <视频1截取的前N帧> <视频2截取的后M帧>"
    echo "示例: $0 100 50"
    exit 1
fi

N=$1
M=$2

VID1="examples/piper_real/logs/04282201/model_input_observation/combined_output.mp4"
VID2="examples/piper_real/logs/04282144/model_input_observation/combined_output.mp4"
OUTPUT="final_combined_output.mp4"

if [ ! -f "$VID1" ] || [ ! -f "$VID2" ]; then
    echo "错误：找不到指定的输入视频文件，请检查路径是否正确。"
    exit 1
fi

echo "正在获取第二个视频的总帧数..."
# 使用 ffprobe 计算视频2的总帧数 (-count_frames 会完整扫描帧数，比较准确)
TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$VID2")

if [ -z "$TOTAL_FRAMES" ]; then
    echo "获取第二个视频的总帧数失败！"
    exit 1
fi

START_FRAME=$((TOTAL_FRAMES - M))
if [ $START_FRAME -lt 0 ]; then
    START_FRAME=0
fi

echo "视频1截取范围: 第 0 帧 到 第 $((N-1)) 帧 (共 $N 帧)"
echo "视频2截取范围: 第 $START_FRAME 帧 到 第 $((TOTAL_FRAMES-1)) 帧 (总帧数 $TOTAL_FRAMES, 截取后 $M 帧)"

# 执行截取与拼接
# 1. trim 过滤器进行精准的帧截取
# 2. setpts=PTS-STARTPTS 将时间戳从0重新对齐
# 3. concat 过滤器连接两个视频流
echo "开始处理并生成视频..."
ffmpeg -y -i "$VID1" -i "$VID2" -filter_complex \
    "[0:v]trim=start_frame=0:end_frame=${N},setpts=PTS-STARTPTS[v1]; \
     [1:v]trim=start_frame=${START_FRAME}:end_frame=${TOTAL_FRAMES},setpts=PTS-STARTPTS[v2]; \
     [v1][v2]concat=n=2:v=1:a=0[outv]" \
    -map "[outv]" -c:v libx264 -pix_fmt yuv420p "$OUTPUT" -loglevel error

echo "拼接完成！输出文件已保存为: $OUTPUT"
