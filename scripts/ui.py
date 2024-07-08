import os
import gradio as gr
import threading
import time
from modules import shared, script_callbacks
# from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
# from modules.ui import  create_sampler_and_steps_selection
# from modules import ui_extra_networks, devices, shared, scripts, script_callbacks, sd_hijack_unet, sd_hijack_utils
import helper.video_util as video_util
from helper.enhance_face import enhance_face
from helper.cn_models import get_cn_models
try:
    from modules.sd_samplers import all_samplers
except ImportError:
    all_samplers = []
try:
    from modules.sd_schedulers import schedulers
except ImportError:
    schedulers = []

def is_queue_enabled():
    # `queue_enabled` 속성이 존재하는지 확인하고, 존재하면 그 값을 반환
    return getattr(shared.demo, 'enable_queue', False)

# Global variable to control the cancellation
share_value = {
    "cancel": False,
    "server_port": 7860,
    "sd_model_checkpoint": "",
}

def extract_video(video, fps, output_path):
    print(f"# video extract mode")
    print(f"video {video}")
    print(f"fps {fps}")
    print(f"output_path {output_path}")
    video_util.extract(video, output_path, fps)    

def combine_video(frames_path, video, fps):
    print(f"# video combine mode")
    print(f"video {video}")
    print(f"fps {fps}")
    print(f"frames_path {frames_path}")
    video_util.combine(frames_path, video, "", fps)    

def ui_buttons(status, complete = False):
    if complete:
        return status, "complete", gr.Button.update(visible=True), gr.Button.update(value="Cancel", visible=False)
    else:
        return status, "", gr.Button.update(visible=False), gr.Button.update(value="Cancel", visible=True)

def process_video_noqueue(
    input_video_mode,upload_video,video_path,input_folder,force_fps,resize_width,resize_height,target_fps,
    seed,sampler,scheduler,sampler_step,cfg_scale,temporalnet_ver,temporalnet_model,temporalnet_weight,prompt,neg_prompt,denoise,
    controlnet_input_image, controlnet_module, controlnet_model, controlnet_weight, controlnet_guidance_start, controlnet_guidance_end,
    output_images_folder,overwrite_output_images,output_video):
    for status, complete, btn1, btn2 in process_video(
        input_video_mode,upload_video,video_path,input_folder,force_fps,resize_width,resize_height,target_fps,
        seed,sampler,scheduler,sampler_step,cfg_scale,temporalnet_ver,temporalnet_model,temporalnet_weight,prompt,neg_prompt,denoise,
        controlnet_input_image, controlnet_module, controlnet_model, controlnet_weight, controlnet_guidance_start, controlnet_guidance_end,
        output_images_folder,overwrite_output_images,output_video):
        print(status)
    
    return ui_buttons(status, True)

def process_video(
    input_video_mode,upload_video,video_path,input_folder,force_fps,resize_width,resize_height,target_fps,
    seed,sampler,scheduler,sampler_step,cfg_scale,temporalnet_ver,temporalnet_model,temporalnet_weight,prompt,neg_prompt,denoise,
    controlnet_input_image, controlnet_module, controlnet_model, controlnet_weight, controlnet_guidance_start, controlnet_guidance_end,
    output_images_folder,overwrite_output_images,output_video):
    global share_value

    # print(controlnet_input_image)
    # print(controlnet_module)
    # print(controlnet_model)
    # print(controlnet_weight)

    from modules.shared_cmd_options import cmd_opts
    share_value["cancel"] = False
    if cmd_opts.port:
        share_value["server_port"] = cmd_opts.port
    share_value["sd_model_checkpoint"] = shared.opts.data["sd_model_checkpoint"]

    yield ui_buttons("prepare..")
    # time.sleep(3)

    if not output_images_folder:
        yield ui_buttons("output_images_folder is empty", True)
        return

    if input_video_mode=="Input Video":
        print(upload_video)
        video_path = upload_video
    
    if input_video_mode=="Input Video" or input_video_mode=="Input Video Path":
        if video_path and os.path.isfile(video_path):
            input_folder = os.path.join(output_images_folder, "video_frames")
            video_fps = extract_video(video_path, force_fps, input_folder)
            if not target_fps:
                target_fps = video_fps
        else:
            print("video file not found")
            yield ui_buttons("video file not found", True)
            return
    
    if not target_fps:
        target_fps = 30

    yield ui_buttons(f"input images : {input_folder} target_fps : {target_fps}")
    time.sleep(1)

    if share_value["cancel"]:
        yield ui_buttons(f"cancel clicked", True)
    
    for update, msg in enhance_face(
        share_value,
        input_folder,
        output_images_folder,
        'fixed',
        seed,
        resize_width,
        resize_height,
        temporalnet_ver,
        temporalnet_model,
        temporalnet_weight,
        prompt,
        neg_prompt,
        sampler,
        scheduler,
        sampler_step,
        cfg_scale,
        denoise,
        overwrite_output_images,
        False,
        0,
        0,
        0,
        controlnet_input_image,
        controlnet_module,
        controlnet_model,
        controlnet_weight, 
        controlnet_guidance_start, 
        controlnet_guidance_end,
    ):
        if update=="done":
            yield ui_buttons("combine video file..")
            if output_video:
                combine_video(output_images_folder, output_video, target_fps)
            else:
                output_video = os.path.join(output_images_folder, "output.mp4")
                combine_video(output_images_folder, os.path.join(output_images_folder, "output.mp4"), target_fps)
            yield ui_buttons(output_video, True)
        elif update=="error":
            yield ui_buttons(msg, True)
        elif update=="cancel":
            yield ui_buttons(msg, True)
        else:
            yield ui_buttons(msg)

def create_controlnet(cn_models):
    with gr.Row(variant="panel"):
        controlnet_input_image = gr.Image(label="Input Image", type="pil", elem_id="input_image")

        with gr.Column(variant="compact"):
            controlnet_module = gr.Dropdown(
                label="ControlNet module",
                choices=["None","ip-adapter_face_id_plus","ip-adapter_face_id"],
                value="None",
                visible=True,
                type="value",
                interactive=True,
            )

            controlnet_model = gr.Dropdown(
                label="ControlNet model",
                choices=cn_models,
                value="None",
                visible=True,
                type="value",
                interactive=True,
            )

            controlnet_weight = gr.Slider(
                label="ControlNet weight",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                visible=True,
                interactive=True,
            )

        with gr.Column(variant="compact"):
            controlnet_guidance_start = gr.Slider(
                label="ControlNet guidance start",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
                visible=True,
                interactive=True,
            )

            controlnet_guidance_end = gr.Slider(
                label="ControlNet guidance end",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                visible=True,
                interactive=True,
            )
    return (controlnet_input_image, controlnet_module, controlnet_model, controlnet_weight, controlnet_guidance_start, controlnet_guidance_end)

def create_temporal_net_util(sampler_names, scheduler_names):
    cn_models = [*get_cn_models()]
    cn_face_models = [*get_cn_models("ip-adapter")]

    with gr.Row():
        with gr.Column(variant='panel'):
            with gr.Row():
                with gr.Tab(label="Input Video") as input_tab1:
                    upload_video = gr.Video(label="Input Video", elem_id="input_video")
                    force_fps = gr.Number(value=0, precision=1, label="Force FPS", interactive=True)
                with gr.Tab(label="Input Video Path") as input_tab2:
                    video_path = gr.Textbox(label="Input Video Path",placeholder="Input Video Path (eg. d:/work/smart.mp4)")
                    force_fps = gr.Number(value=0, precision=1, label="Force FPS", interactive=True)
                with gr.Tab(label="Input Frame Images") as input_tab3:
                    input_folder = gr.Textbox(label="Input Frame Images Folder",placeholder="Input Frame Images Folder (eg. d:/work/smart)")
            with gr.Row(visible=False):
                input_video_mode = gr.Textbox(label="Selected Tab", show_label=False, value="Input Video", interactive=False)
            with gr.Row():
                resize_width = gr.Number(value=0,label="Resize Width", precision=1, interactive=True)
                resize_height = gr.Number(value=0,label="Resize Height", precision=1, interactive=True)
                
        input_tab1.select(lambda :"Input Video", None, input_video_mode)
        input_tab2.select(lambda :"Input Video Path", None, input_video_mode)
        input_tab3.select(lambda :"Input Frame Images", None, input_video_mode)
                    
        with gr.Column(variant='panel'):
            with gr.Tab(label="Face Detail"):
                selected_cn_model = "None"
                for cn_model in cn_models:
                    if cn_model.startswith('diff'):
                        selected_cn_model = cn_model
                        break

                with gr.Row():
                    temporalnet_ver = gr.Dropdown(
                        label="TemporalNet Version",
                        choices=["v1","v2"],
                        value="v1",
                        visible=True,
                    )
                    if cn_models and len(cn_models) > 0:
                        temporalnet_model = gr.Dropdown(
                            label="TemporalNet Models",
                            choices=cn_models,
                            value=selected_cn_model,
                            visible=True,
                        )
                    else:
                        temporalnet_model = gr.Textbox(
                            label="TemporalNet Models",
                            value=selected_cn_model,
                            placeholder="Controlnet model not found. Please paste it directly. (eg. diff_control_sd15_temporalnet_fp16 [adc6bd97])",
                            interactive=True,
                        )

                    temporalnet_weight = gr.Slider(
                        label="TemporalNet Weight",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.40,
                        visible=True,
                        interactive=True,
                    )
                with gr.Row():
                    sampler = gr.Dropdown(
                        label="Sampler",
                        choices=sampler_names,
                        value=sampler_names[0],
                        visible=True,
                    )
                    if len(scheduler_names) > 0:
                        scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=scheduler_names,
                            value=scheduler_names[0],
                            visible=True,
                        )
                    else:
                        scheduler = gr.Textbox(
                            label="Scheduler",
                            value="",
                            visible=False,
                        )
                    seed = gr.Number(value=2223, precision=1, label="Seed", interactive=True)

                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        placeholder="Prompt",
                    )
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        lines=3,
                        placeholder="Negative Prompt",
                    )
                with gr.Row():
                    sampler_step = gr.Number(value=15, precision=1, label="Sampler Step", interactive=True)
                    cfg_scale = gr.Number(value=6, precision=1, label="CFG", interactive=True)
                    denoise = gr.Slider(
                        label="Denoise",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.4,
                        visible=True,
                        interactive=True,
                    )
                
                with gr.Accordion(label="ControlNet (IP-Adapter FaceID)", open=False):
                    controlnet_input_image, controlnet_module, controlnet_model, controlnet_weight, controlnet_guidance_start, controlnet_guidance_end = create_controlnet(cn_face_models)

                with gr.Row():
                    output_images_folder = gr.Textbox(label="Output Images Folder", placeholder="Folder to output converted frame images")
                    overwrite_output_images = gr.Checkbox(label="Overwrite Output Images", placeholder="Overwrite output converted frame images")

                with gr.Row():
                    output_video = gr.Textbox(label="Output Video File (If left blank, it will be automatically saved)", placeholder="Output Video File Path (eg. d:/work/output/smart.mp4)")
                    target_fps = gr.Number(value=30, precision=1, label="Output Video FPS", interactive=True)

                with gr.Row() as queue_evn_buttons:
                    start_button = gr.Button("Run", variant='primary') 
                    cancel_button = gr.Button("Cancel", visible=False)

                with gr.Row() as noqueue_evn_buttons:
                    start_noqueue_button = gr.Button("Run (No Queue)", variant='primary', elem_id="temporalnet_util_start_button") 
                    cancel_noqueue_button = gr.Button("Cancel", elem_id="temporalnet_util_cancel_button")
                with gr.Row() as noqueue_evn_info:
                    gr.Markdown('현재 webui에서 queue를 지원하지 않습니다. 진행상황이 Console 출력으로 표시됩니다. (--no-gradio-queue 옵션이 있다면 빼주세요)')

                with gr.Row(visible=False):
                    update_button = gr.Button("Update", elem_id="temporalnet_util_update_button") 

                with gr.Row():
                    status = gr.Textbox(
                        label="Status",
                        elem_id="temporalnet_util_status",
                        lines=2,
                        interactive=False,
                    )
                with gr.Row(visible=False):
                    check_complete = gr.Textbox(elem_id="temporalnet_util_check_complete", show_label=False, interactive=False)


    def handle_cancel():
        global share_value

        share_value["cancel"] = True
        # time.sleep(3)
        return gr.Button.update(value="Canceling")
    
    def handle_cancel_noqueue():
        global share_value
        share_value["cancel"] = True
        # time.sleep(3)
        return gr.Button.update(value="Cancel")
    
    def update_state():
        temporalnet_util_prompt=shared.opts.data.get("temporalnet_util_prompt", "")
        temporalnet_util_neg_prompt=shared.opts.data.get("temporalnet_util_neg_prompt", "")
        queue_enabled = is_queue_enabled()
        print('[TemporalNetUtil] queue_enabled', queue_enabled)
        if queue_enabled:
            return gr.Row().update(visible=True), gr.Row().update(visible=False), gr.Row().update(visible=False), temporalnet_util_prompt, temporalnet_util_neg_prompt
        else:
            return gr.Row().update(visible=False), gr.Row().update(visible=True), gr.Row().update(visible=True), temporalnet_util_prompt, temporalnet_util_neg_prompt

    update_button.click(fn=update_state,inputs=[],outputs=[queue_evn_buttons,noqueue_evn_buttons,noqueue_evn_info,prompt,neg_prompt])

    cancel_button.click(handle_cancel, inputs=[], outputs=[cancel_button])
    start_button.click(
        fn=process_video, 
        inputs=[
            input_video_mode,upload_video,video_path,input_folder,force_fps,resize_width,resize_height,target_fps,
            seed,sampler,scheduler,sampler_step,cfg_scale,temporalnet_ver,temporalnet_model,temporalnet_weight,prompt,neg_prompt,denoise,
            controlnet_input_image, controlnet_module, controlnet_model, controlnet_weight, controlnet_guidance_start, controlnet_guidance_end,
            output_images_folder,overwrite_output_images,output_video
        ], 
        outputs=[status, check_complete, start_button, cancel_button]
    )

    cancel_noqueue_button.click(handle_cancel_noqueue, inputs=[], outputs=[cancel_button])
    start_noqueue_button.click(
        fn=process_video_noqueue, 
        inputs=[
            input_video_mode,upload_video,video_path,input_folder,force_fps,resize_width,resize_height,target_fps,
            seed,sampler,scheduler,sampler_step,cfg_scale,temporalnet_ver,temporalnet_model,temporalnet_weight,prompt,neg_prompt,denoise,
            controlnet_input_image, controlnet_module, controlnet_model, controlnet_weight, controlnet_guidance_start, controlnet_guidance_end,
            output_images_folder,overwrite_output_images,output_video
        ], 
        outputs=[status, check_complete, start_button, cancel_button]
    )

def on_ui_tabs():
    sampler_names = [sampler.name for sampler in all_samplers]
    scheduler_names = [x.label for x in schedulers]

    with gr.Blocks(analytics_enabled=False) as face_temporal_net:
        create_temporal_net_util(sampler_names, scheduler_names)

        return ((face_temporal_net, "TemporalNet-Util", "TemporalNetUtil"),)

def on_ui_settings():
    section = ('temporalnet_util', "TemporalNet-Util")
    shared.opts.add_option("temporalnet_util_prompt", shared.OptionInfo(
        "(masterpiece, best quality, high quality:1.2) ", "temporalnet_util_prompt", section=section))
    shared.opts.add_option("temporalnet_util_neg_prompt", shared.OptionInfo(
        "(worst quality, low quality:1.3), (angry, mole) ", "temporalnet_util_neg_prompt", section=section))

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)