# TemporalNet Util (stable-diffusion-webui)

TemporalNet을 사용하여 face detail을 지원하는 간단한 sd-webui extension이에요  
(Simple extension that helps with face detail using TemporalNet.)

## Install

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter https://github.com/ninjaneural/temporalnet_util.git to "URL for extension's git repository".
4. Press "Install" button.

## Note

이 확장은 API를 사용하고 있으므로 **--api** 인수를 활성화해야해요  
(This extension is using API, you need to activate the --api argument.)

COMMANDLINE_ARGS에 --api를 추가해주세요~  
(Please add --api to COMMANDLINE_ARGS)  

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

  다운받아서 models/ControlNet 에 저장해주세요!  
  (Please download and save it to models/ControlNet)  

## TemporalNet V2 Guide

