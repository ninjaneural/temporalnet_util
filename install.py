import launch

if not launch.is_installed("ffmpeg"):
    launch.run_pip("install ffmpeg-python", "Install \"ffmpeg-python\" requirements for TemporalNet-Util extension")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "Install \"opencv-python\" requirements for TemporalNet-Util extension")

if not launch.is_installed("ultralytics"):
    launch.run_pip("install ultralytics", "Install \"ultralytics\" requirements for TemporalNet-Util extension")

if not launch.is_installed("huggingface_hub"):
    launch.run_pip("install huggingface_hub", "Install \"huggingface_hub\" requirements for TemporalNet-Util extension")

