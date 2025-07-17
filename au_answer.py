import os
import json
import re
import base64
import requests

# === 可配置路径 ===
VIDEO_ROOT = "/workspace/data/ME_VQA_MEGC_2025_Test/samm_test"
INPUT_JSONL = "/workspace/data/au_detect/me_vqa_samm_test_to_answer.jsonl"  # 你可以替换为实际文件
OUTPUT_JSONL = "/workspace/data/au_detect/me_vqa_samm_test_pred.jsonl"
API_URL = "http://10.1.10.171:3086/upload_image_au"  # 实际接口地址需替换

# === AU 名称映射 ===
au_name_dict = {
    "au1": "inner brow raiser", "au2": "outer brow raiser", "au4": "brow lowerer",
    "au5": "upper lid raiser", "au6": "cheek raiser", "au7": "lid tightener",
    "au9": "nose wrinkler", "au10": "upper lip raiser", "au12": "lip corner puller",
    "au14": "dimpler", "au15": "lip corner depressor", "au16": "lower lip depressor",
    "au17": "chin raiser", "au18": "lip pucker", "au20": "lip stretcher",
    "au23": "lip tightener", "au24": "lip pressor", "au25": "lips part",
    "au26": "jaw drop", "au27": "mouth stretch"
}
au_inv_dict = {v: k for k, v in au_name_dict.items()}

# === AU 阈值字典 ===
au_threshold_dict = {
    "AU1": 0.17, "AU2": 0.14, "AU4": 0.12, "AU5": 0.15, "AU6": 0.55, "AU7": 0.06, "AU9": 0.27,
    "AU10": 0.23, "AU12": 0.21, "AU14": 0.43, "AU15": 0.06, "AU16": 0.06, "AU17": 0.27,
    "AU18": 0.07, "AU20": 0.16, "AU23": 0.05, "AU24": 0.17, "AU25": 0.30, "AU26": 0.33, "AU27": 0.04
}

# === 图像转 Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === 调用 AU 接口
def analyze_au_activation(image_path, api_url, au_threshold_dict, min_diff=0.0):
    payload = {"image": image_to_base64(image_path)}
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        au_result = response.json()[0]["au_res"]
    except Exception as e:
        print(f"API调用失败：{e}")
        return {}

    result = {}
    for au, threshold in au_threshold_dict.items():
        value = round(au_result.get(au, 0.0), 4)
        diff = round(value - threshold, 4)
        activated = diff > min_diff
        strength = diff if activated else 0.0
        result[au] = {
            "value": value,
            "threshold": threshold,
            "activated": activated,
            "strength": round(strength, 4)
        }
    return result

# === 微表情推理函数
def infer_micro_expression(au_result):
    def is_active(au):
        return au_result.get(au, {}).get("activated", False)

    def get_strength(au):
        return au_result.get(au, {}).get("strength", 0.0)

    def avg_strength(au_list):
        active_aus = [au for au in au_list if is_active(au)]
        if not active_aus:
            return 0.0
        return sum(get_strength(au) for au in active_aus) / len(au_list)

    # Step 1: happiness
    if is_active("AU12"):
        return "happiness"

    # Step 2: surprise
    if is_active("AU25"):
        return "surprise"

    # Step 3: anger vs disgust
    au4_active = is_active("AU4")
    au9_active = is_active("AU9")

    if au4_active and not au9_active:
        return "anger"
    elif au4_active and au9_active:
        return "disgust"

    # Step 4: fear vs sadness

    # 判断 (AU1 + AU2 + AU4) 是否全部激活
    group1_active = all(is_active(au) for au in ["AU1", "AU2", "AU4"])
    group2_active = is_active("AU20")

    if not group1_active and not group2_active:
        fear_strength = 0.0
    elif group1_active and group2_active:
        # 两组都激活，取平均值最大者
        fear_strength = max(
            avg_strength(["AU1", "AU2", "AU4"]),
            avg_strength(["AU20"])
        )
    elif group1_active:
        fear_strength = avg_strength(["AU1", "AU2", "AU4"])
    else:  # group2_active
        fear_strength = avg_strength(["AU20"])

    # sadness 仍旧只依赖 AU1
    sadness_strength = avg_strength(["AU1"])

    # 判断最终结果
    if fear_strength == 0 and sadness_strength == 0:
        return "unknown"

    return "fear" if fear_strength >= sadness_strength else "sadness"


# === 最强 AU 提取
def find_strongest_activated_au(au_result):
    max_strength = 0.0
    best_au = None
    for au, info in au_result.items():
        if info["activated"] and info["strength"] > max_strength:
            max_strength = info["strength"]
            best_au = au
    return best_au.lower() if best_au else None

# === 主处理逻辑 ===
output_lines = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        video = item["video"]
        question = item["question"].lower()

        folder = os.path.join(VIDEO_ROOT, video)
        if not os.path.isdir(folder):
            item["answer"] = "video folder not found"
            output_lines.append(item)
            continue

        images = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".bmp"))])
        if not images:
            item["answer"] = "no image found"
            output_lines.append(item)
            continue

        # 处理只有一张图的情况
        if len(images) == 1:
            mid_image = os.path.join(folder, images[0])
        else:
            mid_image = os.path.join(folder, images[len(images) // 2])
        # print(mid_image)
        # 分析 AU
        video_id = item.get("video_id", "unknown")
        au_result = analyze_au_activation(mid_image, API_URL, au_threshold_dict)

        # 获取图像文件名（不含扩展名）作为输出 JSON 文件名的一部分
        image_name = os.path.splitext(os.path.basename(mid_image))[0]
        json_save_path = os.path.join("au_thresholds", f"{video_id}_{image_name}.json")

        # 确保保存目录存在
        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)

        # 保存 AU 结果到对应 JSON 文件
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(au_result, f, indent=2, ensure_ascii=False)

        # 答题逻辑
        if "what is the coarse expression class?" in question:
            micro_expression = infer_micro_expression(au_result)
            if(micro_expression=="happiness"):
                item["answer"] = "positive"
            elif(micro_expression=="surprise"):
                item["answer"] = "surprise"
            else:
                item["answer"] = "negative"
        elif "what is the fine-grained expression class?" in question:
            item["answer"] = infer_micro_expression(au_result)
        elif question.startswith("what is the action unit?"):
            au = find_strongest_activated_au(au_result)
            item["answer"] = au_name_dict.get(au, au) if au else "none"
        elif question.startswith("what are the action units?"):
            active_aus = []
            for au, info in au_result.items():
                if info["strength"] > 0.05:
                    name = au_name_dict.get(au.lower(), au.lower())
                    active_aus.append(name)
            item["answer"] = ", ".join(active_aus) if active_aus else "none"
        elif "is the action unit" in question and "shown" in question:
            match = re.search(r"is the action unit (.+?) shown", question)
            if match:
                au_name = match.group(1).strip().lower()
                au_code = au_inv_dict.get(au_name)
                if au_code:
                    act = au_result.get(au_code.upper(), {}).get("activated", False)
                    item["answer"] = "yes" if act else "no"
                else:
                    item["answer"] = "unknown"
        elif question.startswith("what is the action unit present,"):
            micro_expression = infer_micro_expression(au_result)
            if micro_expression == "happiness":
                coarse_expression = "positive"
            elif micro_expression == "surprise":
                coarse_expression = "surprise"
            else:
                coarse_expression = "negative"

            au_code = find_strongest_activated_au(au_result)
            au_name = au_name_dict.get(au_code, au_code) if au_code else "none"

            item["answer"] = (
                f"The action unit present is: {au_name}. "
                f"Therefore, the fine-grained expression class is {micro_expression}, "
                f"and the coarse expression class is {coarse_expression}."
            )
        elif question.startswith("what are the action units present,"):
            micro_expression = infer_micro_expression(au_result)
            if micro_expression == "happiness":
                coarse_expression = "positive"
            elif micro_expression == "surprise":
                coarse_expression = "surprise"
            else:
                coarse_expression = "negative"

            au_code = find_strongest_activated_au(au_result)
            au_name = au_name_dict.get(au_code, au_code) if au_code else "none"
            active_aus = []
            for au, info in au_result.items():
                if info["strength"] > 0.05:
                    name = au_name_dict.get(au.lower(), au.lower())
                    active_aus.append(name)
            au_names = ", ".join(active_aus) if active_aus else "none"


            item["answer"] = (
                f"The action unit present is: {au_names}. "
                f"Therefore, the fine-grained expression class is {micro_expression}, "
                f"and the coarse expression class is {coarse_expression}."
            )
        else:
            item["answer"] = "unsupported question"

        output_lines.append(item)

# === 写入输出 JSONL ===
with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for item in output_lines:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

