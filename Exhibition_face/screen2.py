import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import os
import logging
import traceback
from collections import deque
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image

mp_face_mesh  = mp.solutions.face_mesh
mp_selfie_seg = mp.solutions.selfie_segmentation

LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
             173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
             263, 466, 388, 387, 386, 385, 384, 398]

SEEDS           = [42, 77, 13, 99, 55, 31, 88, 7, 64, 23]
PROMPT          = "smooth skin, seamless, no eyes, natural face"
NEGATIVE_PROMPT = "eyes, pupils, iris, eyelashes, artifacts, deformed"
SD_STEPS        = 8
SD_SIZE         = 512

SW, SH   = 600, 1024
DST_W, DST_H = 1024, 600

DELAY_FRAMES          = 6
S1_HOLD               = 5.0
S1_TO_S2_FADE         = 6.0
STEP_FADE             = 0.6
S1_GRAIN_DUR          = 2.5   # grain diffusion 지속 시간
PROCESSING_TIMEOUT    = 120.0  # SD가 이 시간 내 완료 안 되면 WAITING 리셋
CAM_RETRY_FRAMES      = 30    # 연속 프레임 실패 후 카메라 재연결 시도
PROCESSING_ENTRY_FADE = 3.3   # grain → S2 첫 프레임 crossfade
S2_TO_S3_FADE         = 3.5   # 단순 crossfade
S3_HOLD               = 3.0
S3_TO_S1_FADE         = 1.5

seed_index            = 0
step_frames           = []
step_lock             = threading.Lock()
os.makedirs("outputs", exist_ok=True)

print("SD 모델 로딩 중...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("mps")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)
print("로딩 완료")

# ── 유틸 ────────────────────────────────────────────

def crop_and_rotate(frame):
    """screen1과 동일 — portrait 크롭 후 90도 회전"""
    fh, fw   = frame.shape[:2]
    target_w = int(fh * SW / SH)
    if target_w > fw:
        target_h = int(fw * SH / SW)
        y0       = (fh - target_h) // 2
        frame    = frame[y0:y0+target_h, :]
    else:
        x0    = (fw - target_w) // 2
        frame = frame[:, x0:x0+target_w]
    frame = cv2.resize(frame, (SW, SH))
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.flip(frame, 0)
    return frame  # → (SH, SW) 회전 후 (1024, 600)

def rotate_mask(mask):
    """segmentation mask도 동일하게 회전"""
    resized = cv2.resize(mask, (SW, SH))
    return cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)

def get_eye_mask(frame, landmarks):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for eye in [LEFT_EYE, RIGHT_EYE]:
        pts = np.array([[int(landmarks[i].x * w),
                         int(landmarks[i].y * h)]
                        for i in eye], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
    mask = cv2.dilate(mask, np.ones((36, 36), np.uint8))
    return mask

def apply_eye_absence(frame, eye_mask):
    blurred = cv2.GaussianBlur(frame, (99, 99), 0)
    blurred = cv2.GaussianBlur(blurred, (51, 51), 0)
    skin    = cv2.mean(frame, mask=eye_mask)[:3]
    solid   = np.full_like(frame, [int(c) for c in skin])
    blurred = cv2.addWeighted(blurred, 0.15, solid, 0.85, 0)
    f1 = cv2.GaussianBlur(eye_mask,(151,151),0).astype(np.float32)/255
    f2 = cv2.GaussianBlur(eye_mask,(101,101),0).astype(np.float32)/255
    f3 = cv2.GaussianBlur(eye_mask,(61, 61), 0).astype(np.float32)/255
    f4 = cv2.GaussianBlur(eye_mask,(31, 31), 0).astype(np.float32)/255
    fe = np.clip(f1*0.4+f2*0.3+f3*0.2+f4*0.1, 0, 1)
    fe = np.power(fe, 0.30)
    fe = np.stack([fe]*3, axis=-1)
    return np.clip(
        frame.astype(np.float32)*(1-fe) +
        blurred.astype(np.float32)*fe, 0, 255
    ).astype(np.uint8)


def fx_thermal_color(frame):
    """열화상 색상 매핑 — 밝기 기준으로 청록→주황→노랑 계열"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # COLORMAP_HSV 기반으로 커스텀 색상 적용
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_HSV)

    # 원본 밝기 정보 유지하면서 색상 입히기
    colored_f = colored.astype(np.float32)
    frame_f   = frame.astype(np.float32)

    # 밝은 영역(얼굴)은 주황/노랑, 어두운 영역은 청록/파랑
    bright_mask = (gray.astype(np.float32) / 255.0)
    bright_mask = np.stack([bright_mask]*3, axis=-1)

    # 주황-노랑 레이어 (밝은 영역)
    warm = np.zeros_like(frame_f)
    warm[:,:,0] = 15                              # B 낮게
    warm[:,:,1] = frame_f[:,:,1] * 0.6 + 15      # G 중간
    warm[:,:,2] = np.clip(frame_f[:,:,2]*1.2+20, 0, 255)  # R

    # 청록 레이어 (어두운 영역)
    cool = np.zeros_like(frame_f)
    cool[:,:,0] = np.clip(frame_f[:,:,0]*1.1+30, 0, 255)  # B
    cool[:,:,1] = np.clip(frame_f[:,:,1]*1.0+12, 0, 255)  # G
    cool[:,:,2] = 10                              # R 낮게

    thermal = warm * bright_mask + cool * (1 - bright_mask)

    # COLORMAP_HSV와 블렌딩
    result = np.clip(thermal*0.7 + colored_f*0.3, 0, 255).astype(np.uint8)
    return result


def fx_thermal_bg(frame):
    """
    배경용 thermal — 인물보다 어둡고 채도 낮게
    desaturate + 약한 thermal 색조 + 강한 blur
    """
    # 강한 blur (depth of field)
    blurred = cv2.GaussianBlur(frame, (61, 61), 0)
    # 채도 낮추기
    hsv          = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1]   = np.clip(hsv[:,:,1]*0.22, 0, 255)   # 채도 낮춤
    hsv[:,:,2]   = np.clip(hsv[:,:,2]*0.93, 0, 255)   # 밝기 살짝 낮춤
    desaturated  = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 약한 thermal 색조 입히기
    thermal_bg   = fx_thermal_color(desaturated)
    # opacity 조절 — 배경 밝기 리프트
    result = cv2.addWeighted(desaturated, 0.35, thermal_bg, 0.45, 15)
    return result


def fx_high_contrast(frame):
    """대비 극대화 — 이목구비 선명하게"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    # L 채널 대비 강화
    l = lab[:,:,0]
    l = np.clip((l - 128) * 1.8 + 128, 0, 255)
    lab[:,:,0] = l
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    # CLAHE로 국소 대비 추가
    clahe  = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    b, g, r = cv2.split(result)
    r = clahe.apply(r)
    g = clahe.apply(g)
    return cv2.merge([b, g, r])

def fx_glow(frame):
    bloom = cv2.GaussianBlur(frame, (31,31), 0)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gray  = np.stack([gray]*3, axis=-1)
    return np.clip(
        frame.astype(np.float32)*(1-gray*0.10) +
        bloom.astype(np.float32)*gray*0.10, 0, 255
    ).astype(np.uint8)

def fx_edge_glow(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(cv2.Canny(gray,40,100),(9,9),0)
    el    = np.zeros_like(frame, dtype=np.float32)
    el[:,:,0] = edges.astype(np.float32)*0.6
    el[:,:,1] = edges.astype(np.float32)*0.3
    el[:,:,2] = edges.astype(np.float32)*0.1
    return np.clip(frame.astype(np.float32)+el*0.25, 0, 255).astype(np.uint8)


def apply_motion_blur_delay(frame_buf, curr):
    """delay feedback으로 모션블러 — 움직임 잔상"""
    if len(frame_buf) < 2:
        return curr
    result = curr.astype(np.float32)
    weights = [0.18, 0.13, 0.09, 0.06, 0.04, 0.02]
    for i, pf in enumerate(reversed(list(frame_buf))):
        if i >= len(weights):
            break
        result = result + pf.astype(np.float32) * weights[i]
    return np.clip(result, 0, 255).astype(np.uint8)

def fx_grain_diffuse(frame, progress):
    """그레인 디퓨전 — melt 잔상이 grain 속으로 흩어지며 PROCESSING으로 연결 (progress 0→1)"""
    img     = frame.astype(np.float32)
    k       = max(1, int(progress * 31))
    if k % 2 == 0: k += 1
    blurred = cv2.GaussianBlur(img.astype(np.uint8), (k, k), 0).astype(np.float32)
    grain   = np.random.normal(0, progress * 55, img.shape).astype(np.float32)
    alpha   = ease_inout(progress)
    return np.clip(img*(1-alpha) + (blurred + grain)*alpha, 0, 255).astype(np.uint8)


def composite_soft(person, bg, person_mask_f):
    soft_mask = cv2.GaussianBlur(person_mask_f, (61, 61), 0)
    soft_mask = np.stack([soft_mask]*3, axis=-1)
    result    = (person.astype(np.float32) * soft_mask
                 + bg.astype(np.float32) * (1 - soft_mask))
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_s1_fx(frame, face_lm, person_mask, frame_buf):
    """frame은 이미 회전된 상태 (DST_H×DST_W)"""
    person_fx = fx_thermal_color(frame)
    person_fx = fx_high_contrast(person_fx)
    person_fx = fx_glow(person_fx)
    person_fx = fx_edge_glow(person_fx)
    if face_lm:
        person_fx = apply_eye_absence(person_fx, get_eye_mask(frame, face_lm))
    person_fx = apply_motion_blur_delay(frame_buf, person_fx)
    frame_buf.append(person_fx.copy())

    # 배경: 실제 카메라 배경(살짝 블러) + thermal 블렌드, 어둡게 해서 인물 강조
    bg_real    = cv2.GaussianBlur(frame, (21, 21), 0)
    bg_thermal = fx_thermal_bg(frame)
    bg_combined = cv2.addWeighted(bg_real, 0.50, bg_thermal, 0.50, 0)
    bg_combined = np.clip(bg_combined.astype(np.float32) * 0.70, 0, 255).astype(np.uint8)

    return composite_soft(person_fx, bg_combined, person_mask)

def ease_inout(t):
    """smoothstep — 시작/끝 부드럽게"""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def crossfade(a, b, alpha):
    a = cv2.resize(a.astype(np.uint8), (b.shape[1], b.shape[0]))
    return cv2.addWeighted(a, 1.0-alpha, b.astype(np.uint8), alpha, 0)

# ── S3 후처리 ────────────────────────────────────────

def make_s3_result(result_bgr, seg_mask):
    """레퍼런스3: 하이키 오버익스포저 — 차가운 톤, 윤곽 유지"""
    img = result_bgr.astype(np.float32)

    # 1) 대비 먼저 — 윤곽/이목구비 보존
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip((lab[:, :, 0] - 128) * 1.5 + 128, 0, 255)
    img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    # 2) 밝기 올리기 (완화된 오버익스포저)
    img = np.clip(img * 1.6 + 20, 0, 255)

    # 3) 쿨 톤 틴팅 (B 올리고 R 낮추기)
    img[:, :, 0] = np.clip(img[:, :, 0] * 1.10 + 15, 0, 255)  # B 올림
    img[:, :, 2] = np.clip(img[:, :, 2] * 0.88,       0, 255)  # R 낮춤

    # 4) 밝은 픽셀만 날리기 (threshold 높여 윤곽 보존)
    lum        = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                              cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cool_white = np.ones_like(img) * np.array([255, 248, 235], dtype=np.float32)
    highlight  = np.clip((lum - 0.70) / 0.25, 0, 1)
    highlight  = np.stack([highlight] * 3, axis=-1)
    img        = img * (1 - highlight) + cool_white * highlight

    # 5) 소프트 포커스 (약하게)
    soft = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (5, 5), 0).astype(np.float32)
    img  = img * 0.85 + soft * 0.15

    # 6) 쿨 블루-그레이 배경 합성
    bg = np.full_like(img, 0)
    bg[:, :, 0] = 225   # B — 차갑게
    bg[:, :, 1] = 218   # G
    bg[:, :, 2] = 208   # R
    mask_f = cv2.GaussianBlur(seg_mask, (61, 61), 0)
    mask_f = np.stack([mask_f] * 3, axis=-1)
    result = img * mask_f + bg * (1 - mask_f)

    return np.clip(result, 0, 255).astype(np.uint8)

# ── S1→S2 전환 — melting ────────────────────────────

def fx_melt_slow(frame, progress, t):
    h, w   = frame.shape[:2]
    amp    = progress * 35
    freq   = 0.018
    map_x  = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y  = np.tile(np.arange(h, dtype=np.float32).reshape(-1,1),(1,w))
    map_x += amp * np.sin(map_y * freq + t * 0.08)
    map_y += amp * 0.5 * np.sin(map_x * freq * 0.5 + t * 0.05)
    drift   = progress * 20
    map_y  += drift * (map_y / h)
    map_x   = np.clip(map_x, 0, w-1).astype(np.float32)
    map_y   = np.clip(map_y, 0, h-1).astype(np.float32)
    warped  = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    k_size  = max(1, int(progress * 25))
    if k_size % 2 == 0: k_size += 1
    kernel  = np.zeros((k_size*2+1, 1), np.float32)
    kernel[k_size:, 0] = np.linspace(0.2, 1.0, k_size+1)
    kernel /= kernel.sum()
    warped  = cv2.filter2D(warped, -1, kernel)
    alpha   = progress * 0.88
    return cv2.addWeighted(frame, 1-alpha, warped, alpha, 0)

# ── S2 denoising 시각화 ──────────────────────────────

def fx_halftone_soft(frame, step_progress):
    dot  = max(3, int(10*(1-step_progress)))
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    step = dot * 2

    Y, X     = np.mgrid[0:h, 0:w]
    cy_grid  = np.clip((Y // step) * step + dot, 0, h - 1)
    cx_grid  = np.clip((X // step) * step + dot, 0, w - 1)
    r_grid   = (gray[cy_grid, cx_grid].astype(np.float32) / 255.0 * dot)
    dist     = np.sqrt((Y - cy_grid) ** 2 + (X - cx_grid) ** 2)
    dot_mask = (dist <= r_grid).astype(np.float32)[:, :, np.newaxis]
    dot_col  = frame[cy_grid, cx_grid].astype(np.float32)
    res      = np.clip(dot_col * dot_mask + frame.astype(np.float32) * (1 - dot_mask),
                       0, 255).astype(np.uint8)

    alpha = 1 - step_progress
    return cv2.addWeighted(frame, 1-alpha*0.5, res, alpha*0.5, 0)

def fx_feedback_soft(prev_buf, curr, step_progress):
    if not prev_buf: return curr
    result = curr.astype(np.float32)
    num    = min(len(prev_buf), 6)
    for i, pf in enumerate(prev_buf[-num:]):
        w = 0.12*(i+1)/num*(1-step_progress*0.6)
        result += pf.astype(np.float32)*w
    return np.clip(result, 0, 255).astype(np.uint8)

def fx_noise_soft(frame, step_progress):
    s = int(22*(1-step_progress))
    if s < 2: return frame
    noise = np.random.normal(0, s, frame.shape).astype(np.float32)
    return np.clip(
        cv2.GaussianBlur(
            np.clip(frame.astype(np.float32)+noise,0,255).astype(np.uint8),
            (3,3),0), 0, 255
    ).astype(np.uint8)

def fx_diffuse_step(frame, step_progress):
    k = max(1, int(11*(1-step_progress)))
    if k%2==0: k+=1
    blurred = cv2.GaussianBlur(frame,(k*2+1,k*2+1),0)
    a       = step_progress**0.6
    return cv2.addWeighted(blurred,1-a,frame,a,0)

def apply_s2_fx(base, step_progress, prev_buf):
    f = fx_diffuse_step(base, step_progress)
    if step_progress < 0.65:
        f = fx_halftone_soft(f, step_progress)
    f = fx_feedback_soft(prev_buf, f, step_progress)
    f = fx_noise_soft(f, step_progress)
    return f

# ── S2→S3 전환 효과 ──────────────────────────────────

def fx_dissolve_warp(a, b, progress, t):
    h, w     = b.shape[:2]
    a        = cv2.resize(a, (w, h))
    amp      = np.sin(progress * np.pi) * 14
    map_x    = np.tile(np.arange(w, dtype=np.float32),(h,1))
    map_y    = np.tile(np.arange(h, dtype=np.float32).reshape(-1,1),(1,w))
    map_x   += amp * np.sin(map_y*0.02 + t*0.06)
    map_y   += amp * 0.4 * np.cos(map_x*0.015 + t*0.04)
    map_x    = np.clip(map_x, 0, w-1).astype(np.float32)
    map_y    = np.clip(map_y, 0, h-1).astype(np.float32)
    warped_a = cv2.remap(a, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    warped_b = cv2.remap(b, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    blended  = cv2.addWeighted(warped_a,1-progress,warped_b,progress,0)
    k = max(1, int((1-abs(progress-0.5)*2)*13))
    if k%2==0: k+=1
    return cv2.GaussianBlur(blended,(k,k),0)

def fx_deform_soft(a, b, progress):
    h, w    = b.shape[:2]
    a       = cv2.resize(a, (w, h))
    amp     = np.sin(progress*np.pi) * 22
    nx      = cv2.GaussianBlur(
        np.random.normal(0,1,(h,w)).astype(np.float32),(61,61),0)*amp
    ny      = cv2.GaussianBlur(
        np.random.normal(0,1,(h,w)).astype(np.float32),(61,61),0)*amp
    map_x   = np.clip(
        np.tile(np.arange(w,dtype=np.float32),(h,1))+nx, 0, w-1)
    map_y   = np.clip(
        np.tile(np.arange(h,dtype=np.float32).reshape(-1,1),(1,w))+ny, 0, h-1)
    warped  = cv2.remap(a, map_x.astype(np.float32),
                        map_y.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    blended = cv2.addWeighted(warped,1-progress,b,progress,0)
    k = max(1,int(np.sin(progress*np.pi)*19))
    if k%2==0: k+=1
    return cv2.GaussianBlur(blended,(k,k),0)

def fx_clarify_soft(a, b, progress):
    h, w    = b.shape[:2]
    a       = cv2.resize(a, (w, h))
    blended = cv2.addWeighted(a,1-progress,b,progress,0)
    k       = max(1,int((1-progress)*31))
    if k%2==0: k+=1
    blurred = cv2.GaussianBlur(blended,(k,k),0)
    return cv2.addWeighted(blurred,1-progress*0.7,blended,progress*0.7,0)

# ── 로깅 ────────────────────────────────────────────
logging.basicConfig(
    filename="exhibition.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s"
)

def reset_to_waiting(reason=""):
    global STATE, state_start, holder, captured_s1, result_s3
    global prev_s2_buf, last_step_i, prev_step_frame, s2_display
    global last_melt_frame, step_frames, frame_buf
    msg = f"리셋 → WAITING: {reason}" if reason else "리셋 → WAITING"
    print(msg)
    logging.warning(msg)
    STATE           = "WAITING"
    state_start     = time.time()
    holder          = {"done": False, "image": None, "result_s3": None, "seed": None}
    captured_s1     = None
    result_s3       = None
    prev_s2_buf     = []
    last_step_i     = -1
    prev_step_frame = None
    s2_display      = None
    last_melt_frame = None
    with step_lock:
        step_frames = []
    frame_buf.clear()

# ── SD 추론 ──────────────────────────────────────────

def decode_latents(pipe, latents):
    with torch.no_grad():
        lat   = latents/pipe.vae.config.scaling_factor
        image = pipe.vae.decode(lat).sample
        image = (image/2+0.5).clamp(0,1)
        image = image.cpu().permute(0,2,3,1).float().numpy()
    return Image.fromarray((image[0]*255).astype(np.uint8))

def run_sd(image_path, mask_path, holder, seed, captured_seg_mask):
    global step_frames
    with step_lock:
        step_frames = []
    try:
        image = Image.open(image_path).convert("RGB").resize((SD_SIZE,SD_SIZE))
        mask  = Image.open(mask_path).convert("L").resize((SD_SIZE,SD_SIZE))

        def step_cb(pipe, step, timestep, kwargs):
            pil = decode_latents(pipe, kwargs["latents"])
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (SW, SH))
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            with step_lock:
                step_frames.append({"step": step, "frame": bgr})
            return kwargs

        result = pipe(
            prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
            image=image, mask_image=mask,
            num_inference_steps=SD_STEPS, guidance_scale=7.5,
            generator=torch.Generator("mps").manual_seed(seed),
            callback_on_step_end=step_cb,
        ).images[0]

        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result_bgr = cv2.resize(result_bgr, (SW, SH))
        result_bgr = cv2.rotate(result_bgr, cv2.ROTATE_90_CLOCKWISE)
        result.save(f"outputs/result_seed{seed}.png")

        result_s3 = make_s3_result(result_bgr, captured_seg_mask)
        cv2.imwrite(f"outputs/result_s3_seed{seed}.png", result_s3)

        holder.update({
            "done":      True,
            "image":     result_bgr,
            "result_s3": result_s3,
            "seed":      seed
        })
        print(f"SD 완료 — seed {seed}")

    except Exception as e:
        msg = f"SD 오류 (seed {seed}): {e}"
        print(msg)
        logging.error(msg + "\n" + traceback.format_exc())
        holder.update({"done": True, "error": str(e),
                       "image": None, "result_s3": None, "seed": seed})

# ── 메인 ────────────────────────────────────────────

cap        = cv2.VideoCapture(0)
face_mesh  = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
selfie_seg = mp_selfie_seg.SelfieSegmentation(model_selection=0)
frame_buf  = deque(maxlen=DELAY_FRAMES)

cv2.namedWindow("installation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("installation", SH, SW)  # 회전 후: 1024×600 → 가로로 넓게

STATE                 = "WAITING"
holder                = {"done":False,"image":None,"result_s3":None,"seed":None}
state_start           = time.time()
captured_s1           = None
result_s3             = None
captured_seg_mask     = np.zeros((SW, SH), dtype=np.float32)
prev_s2_buf           = []
last_step_i           = -1
prev_step_frame       = None
step_transition_start = time.time()
s2_display            = None
last_melt_frame       = None
seed_index            = 0
t                     = 0
cam_fail              = 0   # 연속 카메라 실패 카운터

_cached_person_mask   = np.zeros((DST_H, DST_W), dtype=np.float32)
_cached_face_lm       = None
_ml_frame_count       = 0

print("시작 — q / ESC 종료")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cam_fail += 1
            if cam_fail >= CAM_RETRY_FRAMES:
                logging.warning("카메라 재연결 시도")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(0)
                cam_fail = 0
                reset_to_waiting("카메라 끊김")
            time.sleep(0.033)
            continue
        cam_fail = 0

        try:
            t    += 1
            frame = crop_and_rotate(frame)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            _ml_frame_count += 1
            if _ml_frame_count % 2 == 0:
                seg_r            = selfie_seg.process(rgb)
                raw_mask         = seg_r.segmentation_mask
                _cached_person_mask = cv2.resize(raw_mask, (DST_W, DST_H))
                face_r           = face_mesh.process(rgb)
                _cached_face_lm  = (face_r.multi_face_landmarks[0].landmark
                                    if face_r.multi_face_landmarks else None)

            person_mask   = _cached_person_mask
            face_lm       = _cached_face_lm
            person_mask_b = cv2.GaussianBlur(person_mask, (51, 51), 0)

            s1_out  = apply_s1_fx(frame, face_lm, person_mask_b, frame_buf)
            elapsed = time.time() - state_start
            output  = s1_out   # 모든 state 미매칭 시 기본값

        except Exception as e:
            logging.error(f"프레임 처리 오류: {e}\n{traceback.format_exc()}")
            reset_to_waiting(f"프레임 오류: {e}")
            continue

        # ── WAITING ──────────────────────────────────
        if STATE == "WAITING":
            output = s1_out
            if face_lm:
                STATE       = "S1_HOLD"
                state_start = time.time()

        # ── S1_HOLD ──────────────────────────────────
        elif STATE == "S1_HOLD":
            output = s1_out
            if elapsed >= S1_HOLD:
                # 캡처 시점 마스크 — 이미 DST_H×DST_W 크기
                captured_seg_mask = cv2.resize(person_mask, (SH, SW))  # 회전 후 (1024, 600)

                hard_mask = np.zeros((DST_H, DST_W), dtype=np.uint8)
                if face_lm:
                    hard_mask = get_eye_mask(frame, face_lm)
                    _, hard_mask = cv2.threshold(
                        cv2.GaussianBlur(hard_mask,(51,51),0),
                        64, 255, cv2.THRESH_BINARY)

                # SD 입력용 — 회전 전 원본 비율로 저장
                frame_for_sd = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame_for_sd = cv2.resize(frame_for_sd, (SD_SIZE, SD_SIZE))
                mask_for_sd  = cv2.rotate(hard_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask_for_sd  = cv2.resize(mask_for_sd, (SD_SIZE, SD_SIZE))

                cv2.imwrite("outputs/frame.jpg", frame_for_sd)
                cv2.imwrite("outputs/mask.png",  mask_for_sd)

                captured_s1           = s1_out.copy()
                s2_display            = s1_out.copy()
                prev_s2_buf           = []
                last_step_i           = -1
                prev_step_frame       = None
                step_transition_start = time.time()
                seed                  = SEEDS[seed_index % len(SEEDS)]
                seed_index           += 1
                holder                = {"done":False,"image":None,
                                         "result_s3":None,"seed":None}
                with step_lock:
                    step_frames = []

                threading.Thread(
                    target=run_sd,
                    args=("outputs/frame.jpg","outputs/mask.png",
                          holder, seed, captured_seg_mask.copy()),
                    daemon=True
                ).start()

                STATE       = "S1_TO_S2"
                state_start = time.time()
                print(f"SD 시작 — seed {seed}")

        # ── S1→S2 melting ────────────────────────────
        elif STATE == "S1_TO_S2":
            alpha  = ease_inout(min(elapsed / S1_TO_S2_FADE, 1.0))
            output = fx_melt_slow(s1_out, alpha, t)
            if alpha >= 1.0:
                last_melt_frame = output.copy()
                STATE           = "S1_GRAIN"
                state_start     = time.time()

        # ── Grainy Diffusion ─────────────────────────
        elif STATE == "S1_GRAIN":
            progress = min(elapsed / S1_GRAIN_DUR, 1.0)
            output   = fx_grain_diffuse(last_melt_frame, progress)
            if progress >= 1.0:
                last_melt_frame = output.copy()  # PROCESSING 진입 페이드 시작점
                STATE           = "PROCESSING"
                state_start     = time.time()

        # ── PROCESSING (S2) ──────────────────────────
        elif STATE == "PROCESSING":
            # 타임아웃 또는 SD 오류 → WAITING 리셋
            if elapsed > PROCESSING_TIMEOUT:
                reset_to_waiting(f"SD 타임아웃 ({PROCESSING_TIMEOUT:.0f}초)")
                continue
            if holder.get("error"):
                reset_to_waiting(f"SD 오류: {holder['error']}")
                continue

            with step_lock:
                cur_steps = list(step_frames)

            if cur_steps and len(cur_steps) > last_step_i + 1:
                prev_step_frame = (
                    cur_steps[last_step_i]["frame"].copy()
                    if last_step_i >= 0 and cur_steps[last_step_i]["frame"] is not None
                    else (captured_s1.copy() if captured_s1 is not None
                          else np.zeros((DST_H,DST_W,3),dtype=np.uint8))
                )
                last_step_i          += 1
                step_transition_start = time.time()

            if cur_steps and last_step_i >= 0:
                step_progress = cur_steps[last_step_i]["step"] / SD_STEPS
                base          = cur_steps[last_step_i]["frame"].copy()
                step_elapsed  = time.time() - step_transition_start
                fade_alpha    = ease_inout(min(step_elapsed / STEP_FADE, 1.0))
                if fade_alpha < 1.0 and prev_step_frame is not None:
                    base = crossfade(prev_step_frame, base, fade_alpha)
            else:
                step_progress = 0.0
                base          = (captured_s1.copy()
                                 if captured_s1 is not None
                                 else np.zeros((DST_H,DST_W,3),dtype=np.uint8))

            s2_frame = apply_s2_fx(base, step_progress, prev_s2_buf)
            prev_s2_buf.append(s2_frame.copy())
            if len(prev_s2_buf) > 8:
                prev_s2_buf.pop(0)
            s2_display = s2_frame

            # melt 잔상 → S2 첫 프레임 crossfade (급전환 방지)
            if last_melt_frame is not None:
                entry_alpha = ease_inout(min(elapsed / PROCESSING_ENTRY_FADE, 1.0))
                s2_frame    = crossfade(last_melt_frame, s2_frame, entry_alpha)
                if entry_alpha >= 1.0:
                    last_melt_frame = None

            output = s2_frame

            if holder["done"] and holder["result_s3"] is not None:
                result_s3   = holder["result_s3"]
                STATE       = "S2_TO_S3"
                state_start = time.time()

        # ── S2→S3 crossfade ──────────────────────────────
        elif STATE == "S2_TO_S3":
            raw    = min(elapsed / S2_TO_S3_FADE, 1.0)
            alpha  = ease_inout(raw)
            output = crossfade(s2_display, result_s3, alpha)
            if raw >= 1.0:
                STATE       = "RESULT"
                state_start = time.time()

        # ── RESULT (S3) ───────────────────────────────
        elif STATE == "RESULT":
            output = result_s3
            if elapsed > S3_HOLD:
                STATE       = "S3_TO_S1"
                state_start = time.time()

        # ── S3→S1 ────────────────────────────────────
        elif STATE == "S3_TO_S1":
            alpha  = ease_inout(min(elapsed / S3_TO_S1_FADE, 1.0))
            output = crossfade(result_s3, s1_out, alpha)
            if alpha >= 1.0:
                STATE       = "WAITING"
                state_start = time.time()
                prev_s2_buf = []
                with step_lock:
                    step_frames = []

        cv2.imshow("installation", output)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

finally:
    cap.release()
    face_mesh.close()
    selfie_seg.close()
    cv2.destroyAllWindows()
    print("종료 완료")