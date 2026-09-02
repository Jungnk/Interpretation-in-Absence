import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh  = mp.solutions.face_mesh
mp_selfie_seg = mp.solutions.selfie_segmentation

LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
             173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
             263, 466, 388, 387, 386, 385, 384, 398]

SW, SH         = 600, 1024
DELAY_FRAMES   = 6    # motion blur용 프레임 버퍼 수
BLUR_STRENGTH  = 0.35 # delay 합성 강도

# def make_dark_bg(h, w):
#     """완전 어두운 배경 — 거의 블랙"""
#     bg = np.zeros((h, w, 3), dtype=np.uint8)
#     bg[:, :] = (80, 10, 25)  # 아주 어두운 자주빛
#     return bg


def crop_portrait(frame, SW, SH):
    fh, fw   = frame.shape[:2]
    target_w = int(fh * SW / SH)
    if target_w > fw:
        target_h = int(fw * SH / SW)
        y0       = (fh - target_h) // 2
        frame    = frame[y0:y0+target_h, :]
    else:
        x0    = (fw - target_w) // 2
        frame = frame[:, x0:x0+target_w]
    return cv2.resize(frame, (SW, SH))

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
    """눈 완전 제거 — 매우 강한 blur + 피부톤 + 광범위 feather"""
    blurred = frame.copy()
    for _ in range(10):
        blurred = cv2.GaussianBlur(blurred, (131, 131), 0)

    # 피부톤 평균으로 채움
    skin  = cv2.mean(frame, mask=eye_mask)[:3]
    solid = np.full_like(frame, [int(c) for c in skin])
    blurred = cv2.addWeighted(blurred, 0.15, solid, 0.85, 0)

    # 4단계 feather — 매우 넓게
    f1 = cv2.GaussianBlur(eye_mask, (151,151), 0).astype(np.float32)/255
    f2 = cv2.GaussianBlur(eye_mask, (101,101), 0).astype(np.float32)/255
    f3 = cv2.GaussianBlur(eye_mask, (61, 61),  0).astype(np.float32)/255
    f4 = cv2.GaussianBlur(eye_mask, (31, 31),  0).astype(np.float32)/255
    fe = np.clip(f1*0.4 + f2*0.3 + f3*0.2 + f4*0.1, 0, 1)
    fe = np.stack([fe]*3, axis=-1)

    result = frame.astype(np.float32)*(1-fe) + blurred.astype(np.float32)*fe
    return np.clip(result, 0, 255).astype(np.uint8)

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
    warm[:,:,0] = 30                              # B 낮게
    warm[:,:,1] = frame_f[:,:,1] * 0.6 + 40      # G 중간
    warm[:,:,2] = np.clip(frame_f[:,:,2]*1.5+60, 0, 255)  # R 강하게

    # 청록 레이어 (어두운 영역)
    cool = np.zeros_like(frame_f)
    cool[:,:,0] = np.clip(frame_f[:,:,0]*1.4+80, 0, 255)  # B 강하게
    cool[:,:,1] = np.clip(frame_f[:,:,1]*1.2+40, 0, 255)  # G 중간
    cool[:,:,2] = 20                              # R 낮게

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
    hsv[:,:,1]   = np.clip(hsv[:,:,1]*0.15, 0, 255)   # 채도 많이 낮춤
    hsv[:,:,2]   = np.clip(hsv[:,:,2]*0.85, 0, 255)   # 밝기 많이 낮춤
    desaturated  = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 약한 thermal 색조 입히기
    thermal_bg   = fx_thermal_color(desaturated)
    # opacity 조절 — 배경은 30%만
    result = cv2.addWeighted(desaturated, 0.25, thermal_bg, 0.40, 0)
    return result


def fx_high_contrast(frame):
    """대비 극대화 — 이목구비 선명하게"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    # L 채널 대비 강화
    l = lab[:,:,0]
    l = np.clip((l - 128) * 1.6 + 128, 0, 255)
    lab[:,:,0] = l
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    # CLAHE로 국소 대비 추가
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    b, g, r = cv2.split(result)
    r = clahe.apply(r)
    g = clahe.apply(g)
    return cv2.merge([b, g, r])

def fx_glow(frame):
    # """글로우 — 밝은 영역이 번지는 효과"""
    # bloom1 = cv2.GaussianBlur(frame, (41, 41), 0)
    # bloom2 = cv2.GaussianBlur(frame, (21, 21), 0)
    # result = cv2.addWeighted(frame,  0.65,
    #          cv2.addWeighted(bloom1, 0.25, bloom2, 0.10, 0), 1.0, 0)
    # return np.clip(result, 0, 255).astype(np.uint8)


    bloom = cv2.GaussianBlur(frame, (31, 31), 0)
    # 밝은 영역에만 bloom 적용
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gray  = np.stack([gray]*3, axis=-1)
    glow  = (frame.astype(np.float32) * (1 - gray*0.3)
             + bloom.astype(np.float32) * gray*0.3)
    return np.clip(glow, 0, 255).astype(np.uint8)

def fx_edge_glow(frame):
    """윤곽선 발광 — 얇고 은은하게"""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    edges = cv2.GaussianBlur(edges, (9, 9), 0)
    edge_layer        = np.zeros_like(frame, dtype=np.float32)
    edge_layer[:,:,0] = edges.astype(np.float32) * 0.6   # B
    edge_layer[:,:,1] = edges.astype(np.float32) * 0.3   # G
    edge_layer[:,:,2] = edges.astype(np.float32) * 0.1   # R
    result = frame.astype(np.float32) + edge_layer * 0.25
    return np.clip(result, 0, 255).astype(np.uint8)


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

def composite_soft(person, bg, person_mask_f):
    soft_mask = cv2.GaussianBlur(person_mask_f, (61, 61), 0)
    soft_mask = np.stack([soft_mask]*3, axis=-1)
    result    = (person.astype(np.float32) * soft_mask
                 + bg.astype(np.float32) * (1 - soft_mask))
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_all_fx(frame):
    """전체 효과 파이프라인"""
    f = fx_thermal_color(frame)   # 열화상 색상
    f = fx_high_contrast(f)       # 대비 강화 — 이목구비 선명
    f = fx_glow(f)                # 글로우
    f = fx_edge_glow(f)           # 엣지 발광
    return f

# --- 카메라 ---
cap        = cv2.VideoCapture(0)
face_mesh  = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
selfie_seg = mp_selfie_seg.SelfieSegmentation(model_selection=1)
# bg_frame   = make_dark_bg(SH, SW)

# 프레임 버퍼 (motion blur용)
frame_buf  = deque(maxlen=DELAY_FRAMES)

cv2.namedWindow("S1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("S1", SW, SH)

print("S1 시작 — q / ESC 종료")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_portrait(frame, SW, SH)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 배경 분리
        seg_r       = selfie_seg.process(rgb)
        person_mask = seg_r.segmentation_mask
        person_mask = cv2.GaussianBlur(person_mask, (51, 51), 0)

        # 얼굴 랜드마크
        face_r = face_mesh.process(rgb)

        # 인물 효과
        person_fx = apply_all_fx(frame)

        bg_fx = fx_thermal_bg(frame)

        # 눈 부재
        if face_r.multi_face_landmarks:
            lm        = face_r.multi_face_landmarks[0].landmark
            eye_mask  = get_eye_mask(frame, lm)
            person_fx = apply_eye_absence(person_fx, eye_mask)

        # motion blur delay
        person_fx = apply_motion_blur_delay(frame_buf, person_fx)
        frame_buf.append(person_fx.copy())

        # 배경 합성
        output = composite_soft(person_fx,bg_fx, person_mask)

        cv2.imshow("S1", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

finally:
    cap.release()
    face_mesh.close()
    selfie_seg.close()
    cv2.destroyAllWindows()
    print("종료 완료")