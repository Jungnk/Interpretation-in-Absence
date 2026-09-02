import subprocess
import sys
import os
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "exhibition.log")
HOME = os.path.expanduser("~")

PYTHON_CANDIDATES = [
    f"{HOME}/miniforge3/envs/face-inpainting/bin/python",
    f"{HOME}/miniconda3/envs/face-inpainting/bin/python",
    f"{HOME}/anaconda3/envs/face-inpainting/bin/python",
    f"{HOME}/opt/miniforge3/envs/face-inpainting/bin/python",
]

def find_python():
    for p in PYTHON_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

python = find_python()

if python is None:
    log("오류: face-inpainting conda 환경을 찾을 수 없습니다.")
    log(f"확인한 경로들: {PYTHON_CANDIDATES}")
    input("엔터를 눌러 종료...")
    sys.exit(1)

log(f"Python 경로: {python}")
script = os.path.join(SCRIPT_DIR, "screen2_mini.py")

while True:
    log("Mac Mini 모드 시작")
    result = subprocess.run(
        ["caffeinate", "-d", "-i", python, script],
        cwd=SCRIPT_DIR
    )

    if result.returncode == 0:
        log("정상 종료")
        break

    log(f"비정상 종료 (코드 {result.returncode}) — 5초 후 재시작")
    time.sleep(5)
