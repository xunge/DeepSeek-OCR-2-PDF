import os
import uuid
import zipfile
import shutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import fitz
import io
import torch
from concurrent.futures import ThreadPoolExecutor
import threading


app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_MB", "500")) * 1024 * 1024
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
TEMP_FOLDER = os.getenv("TEMP_FOLDER", "temp")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

task_status = {}
task_lock = threading.Lock()

if torch.version.cuda == "11.8":
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ.setdefault("VLLM_USE_V1", "0")

from config import (
    MODEL_PATH,
    PROMPT,
    SKIP_REPEAT,
    MAX_CONCURRENCY,
    NUM_WORKERS,
    CROP_MODE,
    FLASK_HOST,
    FLASK_PORT,
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.image_process import DeepseekOCR2Processor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from image_caption import describe_image, format_image_description
import re

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

try:
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
    )
    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821,
                                                                                                          128822})]  # window for fast；whitelist_token_ids: <td>,</td>

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    model_loaded = True
except Exception as e:
    print(f"Model loading failed: {e}")
    model_loaded = False


def pdf_to_images(pdf_path, dpi=144):
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    pdf_document.close()
    return images


def prompt_for_single_image(prompt, image_token="<image>"):
    image_token_count = prompt.count(image_token)
    if image_token_count == 0:
        return f"{image_token}\n{prompt}"
    if image_token_count == 1:
        return prompt
    raise ValueError(
        f"PROMPT must contain exactly one {image_token!r} token for single-page OCR, "
        f"but contains {image_token_count}."
    )


def process_single_image(image):
    processor = DeepseekOCR2Processor()
    prompt_in = prompt_for_single_image(PROMPT, processor.image_token)
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {
            "image": processor.tokenize_with_images(
                images=[image], prompt=prompt_in, bos=True, eos=True, cropping=CROP_MODE
            )
        },
    }
    return cache_item


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    matches_image = []
    matches_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_coordinates(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
        return (label_type, cor_list)
    except:
        return None


def draw_bounding_boxes(image, refs, jdx, output_dir):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                color = (
                    np.random.randint(0, 200),
                    np.random.randint(0, 200),
                    np.random.randint(0, 255),
                )
                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            os.makedirs(output_dir, exist_ok=True)
                            cropped.save(f"{output_dir}/{jdx}_{img_idx}.jpg")
                        except:
                            pass
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )

                        text_x = x1
                        text_y = max(0, y1 - 15)
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height],
                            fill=(255, 255, 255, 30),
                        )
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def pil_to_image_bytes(pil_image, quality=95):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="JPEG", quality=quality)
    return img_buffer.getvalue()


def process_pdf_task(task_id, pdf_path, original_filename):
    try:
        with task_lock:
            task_status[task_id] = {
                "status": "processing",
                "progress": 0,
                "message": "正在转换PDF为图片...",
            }

        images = pdf_to_images(pdf_path)

        with task_lock:
            task_status[task_id] = {
                "status": "processing",
                "progress": 20,
                "message": "正在预处理图片...",
            }

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(executor.map(process_single_image, images))

        with task_lock:
            task_status[task_id] = {
                "status": "processing",
                "progress": 40,
                "message": "正在进行OCR识别...",
            }

        outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

        output_dir = os.path.join(OUTPUT_FOLDER, task_id)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        contents_det = ""
        contents = ""
        draw_images = []
        jdx = 0

        with task_lock:
            task_status[task_id] = {
                "status": "processing",
                "progress": 60,
                "message": "正在处理识别结果...",
            }

        for output, img in zip(outputs_list, images):
            content = output.outputs[0].text

            if "<｜end▁of▁sentence｜>" in content:
                content = content.replace("<｜end▁of▁sentence｜>", "")
            else:
                if SKIP_REPEAT:
                    continue

            page_num = f"\n<--- Page Split --->"
            contents_det += content + f"\n{page_num}\n"

            image_draw = img.copy()
            matches_ref, matches_images, matches_other = re_match(content)
            result_image = draw_bounding_boxes(image_draw, matches_ref, jdx, os.path.join(output_dir, "images"))
            draw_images.append(result_image)

            for idx, a_match_image in enumerate(matches_images):
                image_rel_path = f"images/{jdx}_{idx}.jpg"
                image_file_path = os.path.join(output_dir, image_rel_path)
                image_caption = format_image_description(describe_image(image_file_path))
                image_markdown = f"![]({image_rel_path})\n"
                if image_caption:
                    image_markdown += f"\n{image_caption}\n"

                content = content.replace(
                    a_match_image,
                    image_markdown,
                )

            for idx, a_match_other in enumerate(matches_other):
                content = (
                    content.replace(a_match_other, "")
                    .replace("\\coloneqq", ":=")
                    .replace("\\eqqcolon", "=:")
                    .replace("\n\n\n\n", "\n\n")
                    .replace("\n\n\n", "\n\n")
                )

            contents += content + f"\n{page_num}\n"
            jdx += 1

        with task_lock:
            task_status[task_id] = {
                "status": "processing",
                "progress": 80,
                "message": "正在生成结果文件...",
            }

        base_name = os.path.splitext(original_filename)[0]

        with open(
            os.path.join(output_dir, f"{base_name}_det.mmd"), "w", encoding="utf-8"
        ) as f:
            f.write(contents_det)

        with open(
            os.path.join(output_dir, f"{base_name}.mmd"), "w", encoding="utf-8"
        ) as f:
            f.write(contents)

        with task_lock:
            task_status[task_id] = {
                "status": "processing",
                "progress": 90,
                "message": "正在压缩结果...",
            }

        draw_images[0].save(
            os.path.join(output_dir, f"{base_name}_layouts.pdf"),
            save=True,
            append_images=draw_images[1:],
            duration=100,
            loop=0,
        )

        zip_path = os.path.join(OUTPUT_FOLDER, f"{task_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        with task_lock:
            task_status[task_id] = {
                "status": "completed",
                "progress": 100,
                "message": "处理完成",
                "zip_file": f"{task_id}.zip",
            }

        try:
            os.remove(pdf_path)
        except:
            pass

    except Exception as e:
        with task_lock:
            task_status[task_id] = {
                "status": "failed",
                "progress": 0,
                "message": f"处理失败: {str(e)}",
            }
        import traceback

        traceback.print_exc()


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ok" if model_loaded else "error",
            "message": "Model loaded" if model_loaded else "Model not loaded",
        }
    )


@app.route("/api/ocr", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    task_id = str(uuid.uuid4())
    task_status[task_id] = {"status": "pending", "progress": 0, "message": "等待处理"}

    temp_dir = os.path.join(TEMP_FOLDER, task_id)
    os.makedirs(temp_dir, exist_ok=True)

    pdf_files = []
    for file in files:
        if file.filename and file.filename.lower().endswith(".pdf"):
            file_id = str(uuid.uuid4())
            pdf_path = os.path.join(temp_dir, f"{file_id}_{file.filename}")
            file.save(pdf_path)
            pdf_files.append((pdf_path, file.filename))

    if not pdf_files:
        return jsonify({"error": "No valid PDF files"}), 400

    task_thread = threading.Thread(target=process_task_group, args=(task_id, pdf_files))
    task_thread.start()

    return jsonify(
        {
            "task_id": task_id,
            "message": "Files uploaded successfully",
            "status": "pending",
        }
    )


def process_task_group(task_id, pdf_files):
    try:
        with task_lock:
            task_status[task_id] = {
                "status": "processing",
                "progress": 0,
                "message": "开始处理文件...",
            }

        temp_dir = os.path.join(TEMP_FOLDER, task_id)

        total_files = len(pdf_files)
        results = []

        for idx, (pdf_path, filename) in enumerate(pdf_files):
            result_task_id = f"{task_id}_{idx}"
            task_status[result_task_id] = {
                "status": "processing",
                "progress": 0,
                "message": "处理中",
            }

            process_pdf_task(result_task_id, pdf_path, os.path.splitext(filename)[0])

            with task_lock:
                result_status = task_status.get(result_task_id, {})
                if result_status.get("status") == "completed":
                    results.append(
                        {
                            "filename": filename,
                            "status": "completed",
                            "zip_file": result_status.get("zip_file"),
                        }
                    )
                else:
                    results.append(
                        {
                            "filename": filename,
                            "status": "failed",
                            "message": result_status.get("message", "Unknown error"),
                        }
                    )

        output_dir = os.path.join(OUTPUT_FOLDER, task_id)
        os.makedirs(output_dir, exist_ok=True)

        if results:
            combined_zip_path = os.path.join(OUTPUT_FOLDER, f"{task_id}_combined.zip")
            with zipfile.ZipFile(combined_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for result in results:
                    if result["status"] == "completed":
                        zip_file = result["zip_file"]
                        zip_path = os.path.join(OUTPUT_FOLDER, zip_file)
                        if os.path.exists(zip_path):
                            zipf.write(zip_path, os.path.basename(zip_file))
                            os.remove(zip_path)

            if os.path.exists(combined_zip_path):
                with task_lock:
                    task_status[task_id] = {
                        "status": "completed",
                        "progress": 100,
                        "message": f"处理完成 ({len(results)} 个文件)",
                        "zip_file": f"{task_id}_combined.zip",
                        "results": results,
                    }
            else:
                with task_lock:
                    task_status[task_id] = {
                        "status": "completed",
                        "progress": 100,
                        "message": "处理完成",
                        "results": results,
                    }
        else:
            with task_lock:
                task_status[task_id] = {
                    "status": "failed",
                    "progress": 0,
                    "message": "没有有效的PDF文件",
                    "results": results,
                }

        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    except Exception as e:
        with task_lock:
            task_status[task_id] = {
                "status": "failed",
                "progress": 0,
                "message": f"处理失败: {str(e)}",
            }
        import traceback

        traceback.print_exc()


@app.route("/api/status/<task_id>", methods=["GET"])
def get_status(task_id):
    with task_lock:
        status = task_status.get(
            task_id, {"status": "not_found", "message": "Task not found"}
        )
    return jsonify(status)


@app.route("/api/download/<task_id>", methods=["GET"])
def download_result(task_id):
    zip_file = request.args.get("file", "")
    if zip_file:
        file_path = os.path.join(OUTPUT_FOLDER, zip_file)
    else:
        file_path = os.path.join(OUTPUT_FOLDER, f"{task_id}_combined.zip")
        if not os.path.exists(file_path):
            file_path = os.path.join(OUTPUT_FOLDER, f"{task_id}.zip")

    if os.path.exists(file_path):
        return send_file(
            file_path, as_attachment=True, download_name=f"ocr_results_{task_id}.zip"
        )
    else:
        return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
