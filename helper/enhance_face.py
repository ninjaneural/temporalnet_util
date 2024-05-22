import os
import helper.webuiapi_next as webuiapi
from PIL import Image
import numpy as np
from helper.temporalnet2 import make_flow, encode_image
from helper.facedetect import process as face_process
from helper.image_util import resize_image, crop_and_resize, merge_image
from helper.util import get_image_paths
import random

from controlnet_ext import (
    ControlNetExt,
)

cn_ext = ControlNetExt()
if hasattr(cn_ext, 'low_vram'):
    import helper.webuiapi_next as webuiapi
else:
    import helper.webuiapi as webuiapi

unit_tempo_v1 = webuiapi.ControlNetUnit(
    module="none",
    model="diff_control_sd15_temporalnet_fp16 [adc6bd97]",
    weight=1,
    pixel_perfect=True,
)
unit_tempo_v2 = webuiapi.ControlNetUnit(
    module="none",
    model="temporalnetversion2 [b146ac48]",
    weight=1,
    threshold_a=64,
    threshold_b=64,
    pixel_perfect=True,
)

api = webuiapi.WebUIApi()

def enhance_face(cancel_process, input_folder: str, output_folder: str, seed_mode:str, seed:int, frame_width:int, frame_height:int, temporalnet_ver:str, temporalnet_model:str, temporalnet_weight:float, prompt:str, neg_prompt:str, sampler_name:str, sampler_step:int, cfg_scale:int, denoising_strength:float, overwrite: bool, reverse: bool, resume_frame: int, start_frame: int, end_frame: int):
    print(f"###########################################")
    if input_folder and os.path.exists(input_folder):
        face_image_folder = os.path.normpath(os.path.join(output_folder, "./face_images"))
        flow_image_folder = os.path.normpath(os.path.join(output_folder, "./flow_images"))
    else:
        print("input_folder path not found")
        return "error"

    print(f"# input images path {input_folder}")
    print(f"# output path {output_folder}")
    print(f"# denoise {denoising_strength}")
    print(f"# temporalnet_ver {temporalnet_ver}")
    print(f"# temporalnet_model {temporalnet_model}")
    print(f"# temporalnet_weight {temporalnet_weight}")
    print(f"# sampler {sampler_name}")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(face_image_folder, exist_ok=True)
    os.makedirs(flow_image_folder, exist_ok=True)

    if seed == -1:
        seed = random.randrange(1, 2**31)

    print(f"# seed {seed}")

    if not temporalnet_ver:
        temporalnet_ver = "v1"

    # temporalnet v2 bug
    # temporalnet 적용후 controlnet 안쓸때 에러

    face_controlnet_units = []
    # for cn in config.face_controlnet:
    #     cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=config.controlnet_lowvram)
    #     face_controlnet_units.append(cn_unit)

    init_image = None

    input_images_path_list = get_image_paths(input_folder)
    if reverse:
        input_images_path_list.reverse()

    if len(input_images_path_list)==0:
        print("images count : 0")
        return "error"

    input_img = None
    input_img_arr = None

    last_image_arr = None
    flow_image_arr = None

    base_output_image = None
    base_output_image_arr = None

    if start_frame < 1:
        start_frame = 1
    
    if not prompt:
        prompt = "face close up"

    # face config
    face_threshold = 0.2
    face_padding = 16
    face_blur = 16

    start_index = 0
    if init_image != None:
        last_image_arr = np.array(init_image)

    total_frames = len(input_images_path_list)
    print(f"total frames {total_frames}")

    # init frame size
    if frame_width == 0 or frame_height == 0:
        sample_image = Image.open(input_images_path_list[0])
        frame_width = sample_image.width
        frame_height = sample_image.height

    print(f"# frame_width {frame_width}")
    print(f"# frame_height {frame_height}")

    for frame_index in range(start_index, total_frames):                
        output_filename = os.path.basename(input_images_path_list[frame_index])
        output_image_path = os.path.join(output_folder, output_filename)

        frame_number = frame_index + 1
        print(f"# frame {frame_number}/{total_frames}")
        yield f"# frame {frame_number}/{total_frames}"

        if cancel_process.is_set():
            yield "cancel"
            return
        
        if start_frame > frame_number:
            continue

        input_img = Image.open(input_images_path_list[frame_index])
        input_img_arr = np.array(input_img)

        # fit frame size            
        if input_img.width != frame_width or input_img.height != frame_height:
            input_img_arr = resize_image(input_img_arr, frame_width, frame_height, "resize", "center")
            input_img = Image.fromarray(input_img_arr)

        # resume frame
        if frame_number < resume_frame:
            if frame_number == resume_frame - 1:
                last_image_arr = np.array(Image.open(output_image_path))
            continue

        if not overwrite:
            if os.path.isfile(output_image_path):
                print("skip")
                if frame_number < total_frames and not os.path.isfile(os.path.join(output_folder, os.path.basename(input_images_path_list[frame_index + 1]))):
                    print(f"last image {input_images_path_list[frame_index]}")
                    last_image_arr = np.array(Image.open(os.path.join(output_folder, os.path.basename(input_images_path_list[frame_index]))))
                continue

        if temporalnet_ver == "v2":
            if start_frame < frame_number:
                flow_image_arr = make_flow(input_images_path_list[frame_index - 1], input_images_path_list[frame_index], frame_width, frame_height, flow_image_folder, output_filename)

        base_output_image = input_img
        base_output_image_arr = np.array(base_output_image)

        (face_imgs, face_coords, face_mask_arrs) = face_process(input_img_arr, face_threshold, face_padding, face_blur, face_image_folder, output_filename)

        for face_index, (face_img, face_coord, face_mask_arr) in enumerate(zip(face_imgs, face_coords, face_mask_arrs)):
            for unit in face_controlnet_units:
                unit.input_image = face_img

            p_face_controlnet_units = []
            if start_frame == frame_number:
                p_face_controlnet_units = face_controlnet_units
            else:
                unit_tempo = None
                last_face_img_arr = crop_and_resize(last_image_arr, face_coord, face_img.width, face_img.height, frame_width, frame_height)
                if temporalnet_ver == "v2":
                    flow_face_image_arr = crop_and_resize(flow_image_arr, face_coord, face_img.width, face_img.height, frame_width, frame_height)
                    unit_tempo = unit_tempo_v2
                    unit_tempo.model = temporalnet_model
                    unit_tempo.weight = temporalnet_weight
                    unit_tempo.encoded_image = encode_image(flow_face_image_arr, last_face_img_arr)
                elif temporalnet_ver == "v1":
                    unit_tempo = unit_tempo_v1
                    unit_tempo.model = temporalnet_model
                    unit_tempo.weight = temporalnet_weight
                    unit_tempo.input_image = Image.fromarray(last_face_img_arr)
                p_face_controlnet_units = face_controlnet_units + [unit_tempo]

            ret = api.img2img(
                prompt=prompt,
                negative_prompt=neg_prompt,
                sampler_name=sampler_name,
                steps=sampler_step,
                images=[face_img],
                denoising_strength=denoising_strength,
                seed=-1 if seed_mode == "random" else seed if seed_mode == "fixed" else seed + frame_index,
                cfg_scale=cfg_scale,
                width=face_img.width,
                height=face_img.height,
                controlnet_units=[x for x in p_face_controlnet_units if x is not None],
            )

            output_face_filename = f"{os.path.splitext(output_filename)[0]}-face-convert{face_index}.png"
            output_face_image_path = os.path.join(face_image_folder, output_face_filename)
            face_output_image = ret.images[0]
            face_output_image.save(output_face_image_path)
            face_output_image_arr = np.array(face_output_image)

            base_output_image_arr = merge_image(input_img_arr, face_output_image_arr, face_coord, face_mask_arr)

        base_output_image = Image.fromarray(base_output_image_arr)
        base_output_image.save(output_image_path)

        last_image_arr = base_output_image_arr

        if end_frame > 0 and end_frame == frame_number:
            break

    yield "done"