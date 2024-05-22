# TemporalNet Util

TemporalNet을 사용하여 face detail을 지원하는 간단한 extension이에요  
(Simple extension that helps with face detail using TemporalNet.)


## Note

이 확장은 API를 사용하고 있으므로 --api 인수를 활성화해야해요  
(This extension is using API, you need to activate the --api argument.)


* --api 추가하는 예제 windows (webui-user.bat)
```
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--theme dark --xformers --no-half-vae --api

call webui.bat
```

## TemporalNet Models (Controlnet model)

* TemporalNet v1(v3) [(homepage)](https://huggingface.co/CiaraRowles/TemporalNet2)

  [diff_control_sd15_temporalnet_fp16.safetensors Download](https://huggingface.co/CiaraRowles/TemporalNet/resolve/main/diff_control_sd15_temporalnet_fp16.safetensors)

* TemporalNet v2 [(homepage)](https://huggingface.co/CiaraRowles/TemporalNet2)

  [temporalnetversion2.safetensors Download](https://huggingface.co/CiaraRowles/TemporalNet2/resolve/main/temporalnetversion2.safetensors)
