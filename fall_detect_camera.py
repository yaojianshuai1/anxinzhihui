import cv2
import numpy as np
import time
import os
from rknnlite.api import RKNNLite
from PIL import Image, ImageDraw, ImageFont, ImageOps
from collections import deque

RKNN_FILE = '/home/elf/fall_detector.rknn'
MEAN_FILE = '/home/elf/model_mean.npy'
STD_FILE  = '/home/elf/model_std.npy'
IMG_SIZE  = (32, 32)
THRESHOLD = 0.4        # 跌倒判断阈值
CONFIRM_FRAMES = 8     # 连续N帧才确认跌倒

# 输出状态图片（外部无显示器时可通过查看此文件了解检测状态）
OUTPUT_IMAGE = '/home/elf/fall_status.jpg'
# 最小保存间隔（秒），降低频繁写磁盘造成的开销
IMAGE_SAVE_INTERVAL = 0.5

# 用于展示最近识别概率的缓冲（在图片上绘制波形）
RECENT_PROBS = deque(maxlen=120)

def extract_features(frame):
    """从摄像头帧提取1030维特征"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, IMG_SIZE)
    _, binary = cv2.threshold(img_resized, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    geo = np.zeros(6, dtype=np.float32)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        geo = np.array([
            h / (w + 1e-6),
            area / (IMG_SIZE[0] * IMG_SIZE[1]),
            y / IMG_SIZE[1],
            (y + h/2) / IMG_SIZE[1],
            w / IMG_SIZE[0],
            h / IMG_SIZE[1],
        ], dtype=np.float32)

    img_small = cv2.resize(binary, (32, 32)).flatten().astype(np.float32) / 255.0
    return np.concatenate([img_small, geo])





def safe_sigmoid(x):
    # 避免溢出：对极端值进行裁剪
    try:
        import math
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)
    except Exception:
        # 备用实现（numpy）
        import numpy as _np
        return float(1.0 / (1.0 + _np.exp(-_np.clip(x, -50, 50))))

def save_status_image(frame, status, prob, suspect_count, last_save_time):
    """保存灰度状态图，并在图上显示识别过程（灰度缩略图、概率条、最近概率折线）。
    返回新的 last_save_time（时间戳）或原值以维持节流。
    """
    now = time.time()
    if last_save_time != 0 and now - last_save_time < IMAGE_SAVE_INTERVAL:
        return last_save_time

    # 更新最近概率缓冲（线程/进程安全上这里足够）
    try:
        RECENT_PROBS.append(float(prob))
    except Exception:
        pass

    timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

    # 准备灰度基图（确保和训练时一致的黑白风格）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 将灰度图片转换为 PIL，并做自动对比以便在暗场景中可见
    pil_gray = Image.fromarray(gray)
    display_gray = ImageOps.autocontrast(pil_gray, cutoff=1)
    pil_rgb = display_gray.convert('RGB')
    # 在透明 overlay 上绘制 HUD，然后与原图合成，避免覆盖主视图
    overlay = Image.new('RGBA', pil_rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 字体
    font_path_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ]
    font = None
    for fp in font_path_candidates:
        try:
            font = ImageFont.truetype(fp, 20)
            break
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()

    # 文本内容（中文）
    status_text = f'状态: {status}'
    prob_text = f'概率: {prob:.3f}'
    suspect_text = f'疑似帧数: {suspect_count}'
    time_text = f'时间: {timestamp_text}'

    # 绘制文本（白色文字，黑色描边）
    def draw_text_xy(x, y, text):
        draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

    draw_text_xy(10, 10, status_text)
    draw_text_xy(10, 36, prob_text)
    draw_text_xy(10, 62, suspect_text)
    draw_text_xy(10, 88, time_text)

    h, w = gray.shape

    # 绘制灰度缩略图（右上角）
    thumb_w = min(160, w // 3)
    thumb_h = int(thumb_w * h / w)
    thumb = display_gray.resize((thumb_w, thumb_h)).convert('RGB')
    pil_rgb.paste(thumb, (w - thumb_w - 10, 10))

    # 绘制概率条（底部）在 overlay 上绘制半透明概率条，保持原图可见
    bar_h = 12
    bar_w = w - 20
    bar_x = 10
    bar_y = h - bar_h - 10
    # 背景条（半透明）
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(60, 60, 60, 180))
    # 填充比例
    fill_w = int(max(0.0, min(1.0, prob)) * bar_w)
    draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=(200, 200, 200, 220))
    # 标注数值（在 overlay）
    draw.text((bar_x, bar_y - 22), f'Probability: {prob:.3f}', font=font, fill=(255,255,255,230))

    # 绘制最近概率折线（位于概率条上方小区域）
    graph_h = 40
    graph_w = min(300, w - 40)
    graph_x = 10
    graph_y = bar_y - graph_h - 20
    draw.rectangle([graph_x, graph_y, graph_x + graph_w, graph_y + graph_h], outline=(120, 120, 120, 200), fill=(30, 30, 30, 160))
    if len(RECENT_PROBS) >= 2:
        probs = list(RECENT_PROBS)
        # 取最近 graph_w 个点进行绘制（等间隔采样）
        if len(probs) > graph_w:
            step = len(probs) / graph_w
            sampled = [probs[int(i * step)] for i in range(graph_w)]
        else:
            sampled = probs[:]
        # 归一化到图高
        maxp = max(sampled) if sampled else 1.0
        minp = min(sampled) if sampled else 0.0
        span = maxp - minp if maxp != minp else 1.0
        pts = []
        for i, v in enumerate(sampled):
            x = graph_x + int(i * (graph_w / max(1, len(sampled) - 1)))
            y = graph_y + graph_h - int((v - minp) / span * graph_h)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=(220, 220, 220, 230), width=2)

    # 将 overlay 合成到 RGB 图上
    composed = Image.alpha_composite(pil_rgb.convert('RGBA'), overlay)
    # 最终：把合成图转换为灰度保存（保证文件为黑白）
    final_gray = composed.convert('L')
    result = np.array(final_gray)

    try:
        os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
        encode_ok, encimg = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if encode_ok:
            with open(OUTPUT_IMAGE, 'wb') as f:
                f.write(encimg.tobytes())
        else:
            cv2.imwrite(OUTPUT_IMAGE, result)
    except Exception as e:
        print('保存状态图片失败:', e)
    return now


def main():
    # 加载模型
    print('加载RKNN模型...')
    rknn = RKNNLite()
    ret = rknn.load_rknn(RKNN_FILE)
    if ret != 0:
        print('模型加载失败')
        return
    ret = rknn.init_runtime()
    if ret != 0:
        print('NPU初始化失败')
        return
    print('✅ 模型加载成功')

    # 加载归一化参数
    mean = np.load(MEAN_FILE)
    std  = np.load(STD_FILE)

    # 打开摄像头（先尝试/dev/video0，不行换video11或video22）
    cap = cv2.VideoCapture(11)
    if not cap.isOpened():
        print('摄像头打开失败，尝试 /dev/video11...')
        cap = cv2.VideoCapture(11)
    if not cap.isOpened():
        print('摄像头打开失败，请检查连接')
        rknn.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print('✅ 摄像头打开成功')

    # 状态机
    suspect_count = 0
    last_alert_time = 0
    ALERT_COOLDOWN = 60  # 告警冷却60秒
    last_image_save_time = 0

    print('开始检测，按 Ctrl+C 退出...')
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('读取帧失败')
                break

            frame_count += 1

            # 每2帧推理一次（降低CPU占用）
            if frame_count % 2 != 0:
                continue

            # 提取特征
            features = extract_features(frame)
            features = (features - mean) / std
            features = features.reshape(1, 1030).astype(np.float32)

            # NPU推理
            outputs = rknn.inference(inputs=[features])
            # 使用安全 sigmoid 防止溢出
            prob = safe_sigmoid(float(outputs[0][0]))

            # 更新用于显示的概率缓冲
            try:
                RECENT_PROBS.append(float(prob))
            except Exception:
                pass

            # 状态机判断
            if prob > THRESHOLD:
                suspect_count += 1
            else:
                suspect_count = max(0, suspect_count - 2)

            # 确认跌倒
            now = time.time()
            if suspect_count >= CONFIRM_FRAMES:
                if now - last_alert_time > ALERT_COOLDOWN:
                    print(f'🚨 检测到跌倒！概率：{prob:.3f}')
                    last_alert_time = now
                    suspect_count = 0
                    # TODO: 在这里加入告警逻辑（蜂鸣器/推送等）
            else:
                if frame_count % 30 == 0:  # 每30帧打印一次状态
                    status = '疑似跌倒' if suspect_count > 3 else '正常'
                    print(f'状态：{status} | 概率：{prob:.3f} | 疑似帧数：{suspect_count}')

            status = '疑似跌倒' if suspect_count > 3 else '正常'
            last_image_save_time = save_status_image(frame, status, prob, suspect_count, last_image_save_time)

    except KeyboardInterrupt:
        print('\n检测已停止')

    cap.release()
    rknn.release()


if __name__ == '__main__':
    main()
