from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os

# ====== 配置路径 ======
MODEL_PATH = "/workspace/LLaMA-Factory-main/save_models/qwen2.5_vl_7b_sft_model_250624_frame/checkpoint-200/"
INPUT_JSONL = "/workspace/data/7b/me_vqa_casme3_test_to_answer.jsonl"
OUTPUT_JSONL = "/workspace/data/7b/me_vqa_casme3_test_pred.jsonl"
VIDEO_ROOT = "/workspace/data/ME_VQA_MEGC_2025_Test/CAS/"  # 确保最后有斜杠

def get_system_prompt(question):
    q = question.lower()
    if "what is the coarse expression class" in q:
        return (
            "Please analyze the facial expression in the uploaded video.\n\n"
            "Your task is to classify the observed expression into one and only one of the following three coarse expression classes:\n\n"
            "- positive\n"
            "- negative\n"
            "- surprise\n\n"
            "These are the only valid labels.\n"
            # "If the expression seems ambiguous or in between, choose the closest matching of the three listed above.\n\n"
            "You must choose the best-fitting label based solely on the facial cues and emotion conveyed in the video.\n\n"
            "If the expression appears ambiguous or overlaps categories, choose the one that most closely matches the observed expression.\n\n"
            # "Examples:\n\n"
            # "Q: What is the coarse expression class?\n"
            # "A: positive\n\n"
            "Now, answer the following question based on the video:"
        )
    
    elif "what is the fine-grained expression class" in q:
        return (
            "You are a facial expression analysis expert.\n\n" 
            "Please carefully analyze the facial expression in the uploaded video.\n\n"
            "Your task is to classify the observed expression into one and only one of the following eight fine-grained expression categories:\n\n"
            "Valid Expressions to Choose From:\n" 
            "   - anger        \n\n"
            "   - disgust      \n\n"
            "   - fear         \n\n"
            "   - happiness    \n\n"
            "   - sadness      \n\n"
            "   - surprise     \n\n"
            "Only respond with one of the above labels. Do not respond with any label that is not on this list.\n\n"
            "Your answer should be based solely on the facial cues and emotional expressions shown in the video.\n"
            "If the expression appears ambiguous, choose the label that best matches the most dominant emotion displayed.\n\n"
            "Now, based on your analysis of the video, answer with one of the eight labels listed above:"
            # "You are a facial expression analysis expert.\n\n" \
            # "Your task is to analyze the uploaded video and complete the following steps **in order**, based on clearly defined priority rules for Action Units (AUs) and their mapping to fine-grained expressions.\n\n" \
            # "---\n\n" \
            # "**Step 1: Detect AUs**\n" \
            # "Carefully observe the facial motion in the video and identify which Action Units are activated.\n" \
            # "You may only select from the following list of AUs:\n\n" \
            # "   - inner brow raiser\n" \
            # "   - outer brow raiser\n" \
            # "   - brow lowerer\n" \
            # "   - upper lid raiser\n" \
            # "   - cheek raiser\n" \
            # "   - lid tightener\n" \
            # "   - eye closure\n" \
            # "   - eyes left\n" \
            # "   - nose wrinkler\n" \
            # "   - upper lip raiser\n" \
            # "   - lip corner puller\n" \
            # "   - dimpler\n" \
            # "   - lip corner depressor\n" \
            # "   - chin raiser\n" \
            # "   - lip stretcher\n" \
            # "   - lip tightener\n" \
            # "   - lips part\n" \
            # "   - jaw drop\n" \
            # "   - jaw clencher\n" \
            # "   - lip pressor\n" \
            # "   - lower lip depressor\n" \
            # "   - neck tightener\n" \
            # "   - nostril compress\n" \
            # "   - nostril dilater\n" \
            # "   - tilt right\n\n" \
            # "---\n\n" \
            # "**Step 2: Infer Fine-Grained Expression Based on AU Rules (with Priority Order)**\n\n" \
            # "Valid Expressions to Choose From:\n" \
            # "   - anger        \n\n"
            # "   - disgust      \n\n"
            # "   - fear         \n\n"
            # "   - happiness    \n\n"
            # "   - sadness      \n\n"
            # "   - surprise     \n\n"
            # "---\n\n" \
            # "Apply the following rules in sequence, stopping at the first matched condition:\n\n" \
            # "Priority Rule 1: Sadness\n" \
            # "- If both `inner brow raiser (AU1)` and `lip corner depressor (AU15)` are present → Expression is Sadness\n\n" \
            # "Priority Rule 2: Fear vs Surprise\n" \
            # "- If `brow lowerer (AU4)` is present → Expression is Fear\n" \
            # "- Else if `upper lid raiser (AU5)` is present without AU4 → Expression is Surprise\n\n" \
            # "Priority Rule 3: Happiness, Disgust, Repression\n" \
            # "- If `lip corner puller (AU12)` is present → Expression is Happiness\n" \
            # "- Else if `nose wrinkler (AU9)` is present, or both `upper lip raiser (AU10)` and `lip corner depressor (AU15)` are present → Expression is Disgust\n" \
            # "- Else if both `brow lowerer (AU4)` and `lid tightener (AU7)` are present → Expression is Repression\n\n" \
            # "If none of the above rules match, apply your expert judgment to select the most appropriate expression from the Valid Expressions list.\n\n" \
            # "---\n\n" \

            # "Your output must follow this format:\n\n" \
            # "[one expression from the list above]\n" \
        )
    elif "shown on the face" in q:
        return (
            "Please analyze the uploaded video.\n\n"
            "For the following question, respond with “yes” or “no” depending on whether the specific action unit is clearly visible in the video.\n\n"
            "The action unit will be one of the following:\n\n"
            "   - inner brow raiser\n"
            "   - outer brow raiser\n"
            "   - brow lowerer\n"
            "   - upper lid raiser\n"
            "   - cheek raiser\n"
            "   - lid tightener\n"
            "   - eye closure\n"
            "   - eyes left\n"
            "   - nose wrinkler\n"
            "   - upper lip raiser\n"
            "   - lip corner puller\n"
            "   - dimpler\n"
            "   - lip corner depressor\n"
            "   - chin raiser\n"
            "   - lip stretcher\n"
            "   - lip tightener\n"
            "   - lips part\n"
            "   - jaw drop\n"
            "   - jaw clencher\n"
            "   - lip pressor\n"
            "   - lower lip depressor\n"
            "   - neck tightener\n"
            "   - nostril compress\n"
            "   - nostril dilater\n"
            "   - tilt right\n\n"
            "Do not provide explanation. Just return  “yes” or “no”.\n\n"
            "Now, answer the following question based on the video: "
        )
    elif "located on the left or right face" in q:
        return (
            "Please analyze the uploaded video.\n\n"
            "For the following question, determine whether the given action unit appears on the **left** or **right** side of the face.\n\n"
            "The action unit will be one of the following:\n\n"
            "   - inner brow raiser\n"
            "   - outer brow raiser\n"
            "   - brow lowerer\n"
            "   - upper lid raiser\n"
            "   - cheek raiser\n"
            "   - lid tightener\n"
            "   - eye closure\n"
            "   - eyes left\n"
            "   - nose wrinkler\n"
            "   - upper lip raiser\n"
            "   - lip corner puller\n"
            "   - dimpler\n"
            "   - lip corner depressor\n"
            "   - chin raiser\n"
            "   - lip stretcher\n"
            "   - lip tightener\n"
            "   - lips part\n"
            "   - jaw drop\n"
            "   - jaw clencher\n"
            "   - lip pressor\n"
            "   - lower lip depressor\n"
            "   - neck tightener\n"
            "   - nostril compress\n"
            "   - nostril dilater\n"
            "   - tilt right\n\n"
            "Use only “left” or “right” as your answer. Do not include both or any explanation.\n\n"
            "Now, answer the following question based on the video: "
        )
    elif "what is the action unit present" in q \
       and "fine-grained expression" in q \
       and "coarse expression class" in q:
        # 单一AU，细粒度+粗粒度表情提示词
        return (
            "Please analyze the uploaded video.\n\n"
            "Your task is to determine the following three elements based on the facial expressions shown in the video:\n\n"
            "1. The **action units (AUs)** present in the video. You must list **one* AUs, selected only from the following list:\n\n"
            "   - inner brow raiser\n"
            "   - outer brow raiser\n"
            "   - brow lowerer\n"
            "   - upper lid raiser\n"
            "   - cheek raiser\n"
            "   - lid tightener\n"
            "   - eye closure\n"
            "   - eyes left\n"
            "   - nose wrinkler\n"
            "   - upper lip raiser\n"
            "   - lip corner puller\n"
            "   - dimpler\n"
            "   - lip corner depressor\n"
            "   - chin raiser\n"
            "   - lip stretcher\n"
            "   - lip tightener\n"
            "   - lips part\n"
            "   - jaw drop\n"
            "   - jaw clencher\n"
            "   - lip pressor\n"
            "   - lower lip depressor\n"
            "   - neck tightener\n"
            "   - nostril compress\n"
            "   - nostril dilater\n"
            "   - tilt right\n\n"
            "2. The corresponding **fine-grained expression class**, chosen from:  \n"
            "   - anger        \n\n"
            "   - disgust      \n\n"
            "   - fear         \n\n"
            "   - happiness    \n\n"
            "   - sadness      \n\n"
            "   - surprise     \n\n"
            "3. The corresponding **coarse expression class**, chosen from:  \n"
            "   - positive  \n"
            "   - negative  \n"
            "   - surprise\n\n"
            "Please answer in the following format by replacing placeholders with your predicted values.\n\n"
            "Format:\n"
            "The action units present are: <AU>. Therefore, the fine-grained expression class is <fine>, and the coarse expression class is <coarse>.\n\n"
            "Important notes:\n\n"
            "- Only choose **one** AU, **one** fine-grained expression, and **one** coarse expression.\n"
            "- Do **not** include any explanations or additional text.\n"
            "- Use the **exact names** from the provided lists (no paraphrasing or synonyms).\n\n"
            "Now, answer the following question based on the video: "
        )

    elif "what are the action units present" in q \
         and "fine-grained expression" in q \
         and "coarse expression class" in q:
        # 多个AU，细粒度+粗粒度表情提示词
        return (
            "Please analyze the uploaded video.\n\n"
            "Your task is to determine the following three elements based on the facial expressions shown in the video:\n\n"
            "1. The **action units (AUs)** present in the video. You must list **two or more** AUs, selected only from the following list:\n\n"
            "   - inner brow raiser\n"
            "   - outer brow raiser\n"
            "   - brow lowerer\n"
            "   - upper lid raiser\n"
            "   - cheek raiser\n"
            "   - lid tightener\n"
            "   - eye closure\n"
            "   - eyes left\n"
            "   - nose wrinkler\n"
            "   - upper lip raiser\n"
            "   - lip corner puller\n"
            "   - dimpler\n"
            "   - lip corner depressor\n"
            "   - chin raiser\n"
            "   - lip stretcher\n"
            "   - lip tightener\n"
            "   - lips part\n"
            "   - jaw drop\n"
            "   - jaw clencher\n"
            "   - lip pressor\n"
            "   - lower lip depressor\n"
            "   - neck tightener\n"
            "   - nostril compress\n"
            "   - nostril dilater\n"
            "   - tilt right\n\n"
            "2. The corresponding **fine-grained expression class**, chosen from:  \n"
            "   - anger        \n\n"
            "   - disgust      \n\n"
            "   - fear         \n\n"
            "   - happiness    \n\n"
            "   - sadness      \n\n"
            "   - surprise     \n\n"
            "3. The corresponding **coarse expression class**, chosen from:  \n"
            "   - positive  \n"
            "   - negative  \n"
            "   - surprise\n\n"
            "- You must list at least two AUs, but the total number of AUs can vary depending on what is present in the video.\n"
            "- Do not include any explanations.  \n"
            "- Do not invent new AUs. Only use AUs from the list.  \n"
            "Please answer in the following format by replacing all placeholders (e.g., <AU1>, <fine>, <coarse>) with your predicted values.\n\n"
            "Format:\n"
            "The action units present are: <AU1>, <AU2>, ..., <AUn>. Therefore, the fine-grained expression class is <fine>, and the coarse expression class is <coarse>.\n\n"
            # "Examples:\n\n"
            # "Q: What are the action units present, and based on them, what is the fine-grained expression and the coarse expression class?  \n\n"
            # "A: The action units present are: brow lowerer, upper lip raiser. Therefore, the fine-grained expression class is disgust, and the coarse expression class is negative.\n\n"
            "Now, answer the following question based on the video: "
        )
    elif "what is the action unit" in q:
        return (
            "Please analyze the uploaded video.\n\n"
            "For the following question, your answer should be **only one** action unit that is most clearly present in the video.\n\n"
            "You must choose from the following action units:\n\n"
            # "STRICT INSTRUCTIONS:\n"
            # "- You must choose exactly ONE AU from the list.\n"
            # "- You are not allowed to make up or guess AUs that are not in the list.\n"
            # "- The AU must be written exactly as it appears below.\n"
            "Valid Action Units (AUs):\n"
            "- inner brow raiser\n"
            "- outer brow raiser\n"
            "- brow lowerer\n"
            "- upper lid raiser\n"
            "- cheek raiser\n"
            "- lid tightener\n"
            "- eye closure\n"
            "- eyes left\n"
            "- nose wrinkler\n"
            "- upper lip raiser\n"
            "- lip corner puller\n"
            "- dimpler\n"
            "- lip corner depressor\n"
            "- chin raiser\n"
            "- lip stretcher\n"
            "- lip tightener\n"
            "- lips part\n"
            "- jaw drop\n"
            "- jaw clencher\n"
            "- lip pressor\n"
            "- lower lip depressor\n"
            "- neck tightener\n"
            "- nostril compress\n"
            "- nostril dilater\n"
            "- tilt right\n\n"
            "Only respond with one AU from this list. Do not add any explanation or additional AUs.\n\n"
            "Now, answer the following question based on the video: "
        )
    elif "what are the action units" in q:
        return (
            "Please analyze the uploaded video.\n\n"
            "For the following question, list **all the action units** that are clearly shown in the video.\n\n"
            "You must choose from the following action units:\n\n"
            # "STRICT INSTRUCTIONS:\n"
            # "- Do NOT invent or infer AUs not listed below.\n"
            # "- You MUST choose at least ONE AU from the list.\n"
            # "- If no AU is clearly shown, choose the one you think is the most likely.\n"
            "- Do NOT include anything outside the list.\n"
            "Valid Action Units (AUs):\n"
            "- inner brow raiser\n"
            "- outer brow raiser\n"
            "- brow lowerer\n"
            "- upper lid raiser\n"
            "- cheek raiser\n"
            "- lid tightener\n"
            "- eye closure\n"
            "- eyes left\n"
            "- nose wrinkler\n"
            "- upper lip raiser\n"
            "- lip corner puller\n"
            "- dimpler\n"
            "- lip corner depressor\n"
            "- chin raiser\n"
            "- lip stretcher\n"
            "- lip tightener\n"
            "- lips part\n"
            "- jaw drop\n"
            "- jaw clencher\n"
            "- lip pressor\n"
            "- lower lip depressor\n"
            "- neck tightener\n"
            "- nostril compress\n"
            "- nostril dilater\n"
            "- tilt right\n\n"
            "Only respond with the AU names, separated by commas if there are multiple. Do not invent any AUs that are not in this list.\n\n"
            "Now, answer the following question based on the video: "
        )
    else:
        # 默认提示词
        return "You are a helpful assistant."


# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video1.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "file:///path/to/frame1.jpg",
                    "file:///path/to/frame2.jpg",
                    "file:///path/to/frame3.jpg",
                    "file:///path/to/frame4.jpg",
                ],
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]



# ====== 主逻辑：读取输入、处理、写入输出 ======
with open(INPUT_JSONL, "r", encoding="utf-8") as fin, open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line.strip())
        question = item["question"]
        folder_name = item["video"]  # 例如 "SAMM-3"
        folder_path = os.path.join(VIDEO_ROOT, folder_name)

        # 获取所有图像路径并排序（保证顺序一致）
        if not os.path.isdir(folder_path):
            print(f"Warning: 文件夹不存在 - {folder_path}")
            continue

        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        # 构造 file:/// 路径
        image_paths = [f"file:///{os.path.abspath(os.path.join(folder_path, img))}" for img in image_files]

        # 构造系统提示和 messages
        system_prompt = get_system_prompt(question)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": image_paths,
                    },
                    {"type": "text", "text": question},
                ]
            }
        ]
        # print(messages)

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
         # 写入结果
        item["answer"] = output_text[0]
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ Done: {item['video_id']} => {output_text[0]}")
