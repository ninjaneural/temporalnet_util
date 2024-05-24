import os
import ffmpeg
import re

def nat_sort(arr):
    def split_string(s):
        processed_name = []
        i = 0
        while i < len(s):
            is_num = s[i].isdigit()
            j = i + 1
            while j < len(s) and s[j].isdigit() == is_num:
                j += 1
            processed_name.append(s[i:j])
            i = j
        return processed_name

    def compare(a, b):
        len_a = len(a)
        len_b = len(b)
        min_len = min(len_a, len_b)

        for i in range(min_len):
            if a[i] != b[i]:
                is_num_a = a[i].isdigit()
                is_num_b = b[i].isdigit()
                if is_num_a and is_num_b:
                    return int(a[i]) - int(b[i])
                elif is_num_a:
                    return -1
                elif is_num_b:
                    return 1
                else:
                    return -1 if a[i] < b[i] else 1
        
        return len_a - len_b

    split_arr = [split_string(v) for v in arr]
    split_arr.sort(key=lambda x: (x, compare))
    return [''.join(v) for v in split_arr]

def get_video_fps(video_path):
    try:
        # ffmpeg.probe를 사용하여 비디오 파일의 메타데이터를 가져옴
        probe = ffmpeg.probe(video_path)
        # 비디오 스트림 정보를 찾음
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            return 0
        
        # 평균 프레임 속도 가져오기
        avg_frame_rate = video_stream['avg_frame_rate']
        # 분자와 분모를 분리하여 FPS 계산
        numerator, denominator = map(int, avg_frame_rate.split('/'))
        fps = numerator / denominator
        return fps
    except:
        return 0

def extract(video_path, output_dir, fps=0, format="%07d.png"):
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, format)

    video_fps = get_video_fps(video_path)
    
    if fps:
        (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=fps)
            .output(output_template)
            .run()
        )
    else:
        (
            ffmpeg
            .input(video_path)
            .output(output_template)
            .run()
        )
        
    return video_fps

def combine(images_dir, output_path, sound_video_path="", fps=15, crf=17):
    # 자연 정렬을 사용하여 이미지 파일 리스트 생성
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    image_files = nat_sort(image_files)
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    
    # ffmpeg concat demuxer input 파일 생성
    concat_input_file = os.path.join(images_dir, 'concat_input.txt')
    with open(concat_input_file, 'w') as f:
        for image_path in image_paths:
            f.write(f"file '{image_path}'\n")
    
    video_input = ffmpeg.input(concat_input_file, r=fps, f='concat', safe=0)

    video_output_args = {
        'c:v': 'libx264',
        'pix_fmt': 'yuv420p',
        'crf': crf
    }

    if sound_video_path:
        audio_input = ffmpeg.input(sound_video_path)
        (
            ffmpeg
            .output(video_input, audio_input, output_path, 
                    **video_output_args,
                    vcodec='libx264', acodec='copy', pix_fmt='yuv420p', crf=crf)
            .run(overwrite_output=True)
        )
    else:
        (
            ffmpeg
            .output(video_input, output_path, **video_output_args)
            .run(overwrite_output=True)
        )
