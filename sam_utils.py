import gradio as gr
import numpy as np
import torch
import os
import sam3
import gc
import ctypes
from PIL import Image, ImageFilter

def free_system_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


try:
    from sam3 import build_sam3_image_model
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

    SAM3_AVAILABLE = True
except Exception as e:
    print(e)
    SAM3_AVAILABLE = False

global_model = None
global_processor = None
bpe_path = None

if SAM3_AVAILABLE:
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

cached_masks = []


def load_sam3_model():
    """Load the model into VRAM only when needed"""
    global global_model, global_processor
    if not SAM3_AVAILABLE or bpe_path is None:
        return False

    if global_model is None:
        print("Loading SAM3 into VRAM...")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            global_model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path="sam3.1_multiplex.pt")
            global_processor = Sam3Processor(global_model, confidence_threshold=0.5)
        print("SAM3 loaded successfully!")
    return True


def unload_sam3_model():
    """Completely unload SAM3 from VRAM to make room for SD1.5"""
    global global_model, global_processor, cached_masks

    global_model = None
    global_processor = None

    free_system_memory()
    print("SAM3 completely removed from VRAM.")
    return "SAM3 unloaded from VRAM! You can now start Stable Diffusion 1.5."


def generate_sam3_masks(input_image_editor, text_prompt):
    global cached_masks, global_model, global_processor

    cached_masks.clear()
    free_system_memory()

    if input_image_editor is None:
        return gr.update(), gr.update(choices=[], value=[]), "Load an image first."

    image_data = input_image_editor.get("background")
    if image_data is None:
        return gr.update(), gr.update(choices=[], value=[]), "No image found."

    if not isinstance(image_data, np.ndarray):
        image_data = np.array(image_data)
    image = Image.fromarray(image_data).convert("RGB")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if load_sam3_model():
            inference_state = None
            try:
                inference_state = global_processor.set_image(image)
                global_processor.reset_all_prompts(inference_state)
                inference_state = global_processor.set_text_prompt(state=inference_state, prompt=text_prompt)

                masks_tensor = inference_state.get("masks", [])
                for i in range(len(masks_tensor)):
                    mask_np = masks_tensor[i].permute(1, 2, 0).detach().cpu().numpy()[..., 0]
                    cached_masks.append(mask_np)
            finally:
                if inference_state is not None:
                    del inference_state
                free_system_memory()
        else:
            w, h = image.size
            m1, m2, m3 = np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)
            m1[h // 4:h // 2, w // 4:w // 2] = True
            m2[h // 2:3 * h // 4, w // 2:3 * h // 4] = True
            m3[h // 3:2 * h // 3, w // 3:2 * h // 3] = True
            cached_masks = [m1, m2, m3]

    choices = [f"Mask {i + 1}" for i in range(len(cached_masks))]
    default_selection = [choices[0]] if choices else []

    initial_editor_data = update_editor_layers(image, default_selection)
    unload_sam3_model()
    return initial_editor_data, gr.update(choices=choices, value=default_selection), "Masks generated!"


def update_editor_layers(image, selected_masks_names):
    global cached_masks
    w, h = image.size
    combined_mask = np.zeros((h, w), dtype=bool)

    for name in selected_masks_names:
        try:
            idx = int(name.split(" ")[1]) - 1
            if 0 <= idx < len(cached_masks):
                combined_mask = np.logical_or(combined_mask, cached_masks[idx])
        except (IndexError, ValueError):
            continue

    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[combined_mask] = [0, 0, 255, 160]

    return {
        "background": image,
        "layers": [mask_rgba],
        "composite": image
    }


def on_checkbox_change(selected_masks, input_image_editor):
    if input_image_editor is None:
        return None
    image_data = input_image_editor.get("background")
    if image_data is None:
        return None
    if not isinstance(image_data, np.ndarray):
        image_data = np.array(image_data)
    image = Image.fromarray(image_data).convert("RGB")
    return update_editor_layers(image, selected_masks)


def process_corrected_mask(
    editor_output,
    dilation_radius,
    blur_radius
):
    if editor_output is None or not editor_output.get("layers"):
        return "Nessuna maschera rilevata da salvare.", None

    mask_layer = np.array(editor_output["layers"][0])

    # Extract alpha channel
    final_binary_mask = (
        mask_layer[:, :, 3] > 0
    ).astype(np.uint8) * 255

    mask_img = Image.fromarray(final_binary_mask).convert("L")

    # Dilation
    if dilation_radius > 0:
        kernel_size = dilation_radius * 2 + 1

        # PIL MaxFilter acts as a mask expansion
        if kernel_size % 2 == 0:
            kernel_size += 1

        mask_img = mask_img.filter(
            ImageFilter.MaxFilter(kernel_size)
        )

    # Feather edges
    if blur_radius > 0:
        mask_img = mask_img.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )

    mask_path = "/tmp/tmp_mask.png"
    mask_img.save(mask_path)

    return (
        f"Maschera generata (dilate={dilation_radius}, blur={blur_radius})",
        mask_path
    )


def swap_dimensions(w, h):
    return h, w


def resize_and_show(img, mask, w, h):
    if img is None or mask is None:
        return None, None

    try:
        width, height = int(w), int(h)
    except ValueError:
        return img, mask

    image_data = img.get("background")
    if image_data is None:
        return None, None

    if not isinstance(image_data, np.ndarray):
        image_data = np.array(image_data)
    image = Image.fromarray(image_data).convert("RGB")

    with Image.open(mask) as mask_img:
        resized_img = image.resize((width, height))
        resized_mask = mask_img.resize((width, height))

    free_system_memory()
    return resized_img, resized_mask