import os
import ffmpeg

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

def combine(images_dir, output_path, sound_video_path="", fps=15, format="%07d.png", start_number=1, crf=17):
    input_pattern = os.path.join(images_dir, format)
    
    video_input = (
        ffmpeg
        .input(input_pattern, r=fps, start_number=start_number)
    )

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
