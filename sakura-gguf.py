import pyaudio
import numpy as np
import threading
import queue
import torch
import time
from datetime import datetime
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from tkinter import filedialog
import json
import os
import resampy
import soundfile as sf
import gc
from PIL import Image, ImageDraw, ImageFont
import textwrap
import cv2
import numpy as np
import requests, base64

import webrtcvad

from demucs.pretrained import get_model as demucs_get_model
from demucs.apply import apply_model as demucs_apply_model

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

from llama_cpp import Llama

qwen_gguf_path = "./sakura-7b-qwen2.5-v1.0-q6k.gguf"    # sakura GGUF 模型路徑
hf_whisper_path = "./turbo"                             # Whisper large-turbo 模型路徑
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5vl:7b-q4_K_M"                    # ollama模型名稱 qwen2.5vl:3b-q4_K_M or qwen2.5vl:7b-q4_K_M

os.environ["OMP_NUM_THREADS"] = "1"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "translated_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)
# ---------- 視窗固定 ----------
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  
    # 2 => PROCESS_PER_MONITOR_DPI_AWARE
    # 1 => SYSTEM_DPI_AWARE
except:
    pass

MODE_VOICE = "語音轉錄"
MODE_OCR = "螢幕辨識"
current_mode = MODE_VOICE  # 默認語音轉錄

# 標記是否已加載語音相關模型（初始只加載 LLM）
voice_models_loaded = False

vlm_active = False   # 勾選並且啟動過 VLM 流程後設為 True

paddleocr_model = None  # 全局變量
paddleocr_lang = None
persistent_window = None  # 記錄持久OCR窗口對象
persistent_ocr_active = False  # 標記是否正在進行持續OCR
ocr_help_shown = False
def load_paddleocr(lang="japan"):
    global paddleocr_model, paddleocr_lang
    from paddleocr import PaddleOCR
    need_init = (paddleocr_model is None) or (paddleocr_lang != lang)
    if need_init:
        first_time = (paddleocr_model is None)
        paddleocr_model = PaddleOCR(use_angle_cls=True, lang=lang)
        paddleocr_lang = lang
        update_unload_button_state()
        
record_thread = None
transcribe_thread = None

# ---------- 釋放 / 按鈕開關 ----------
def update_unload_button_state():
    """
    根據模型是否已載入來開關 手動卸載 按鈕
    """
    has_stt = bool(voice_models_loaded)
    has_ocr = (paddleocr_model is not None)
    has_vlm = bool(vlm_active)

    if has_stt or has_ocr or has_vlm:
        unload_btn.config(state="normal")
    else:
        unload_btn.config(state="disabled")

def unload_models():
    """
    釋放 Whisper / Demucs 與 PaddleOCR 佔用的資源
    """
    global hf_model, hf_processor, stt_pipe, demucs_model, voice_models_loaded
    global paddleocr_model
    global vlm_active
    # Whisper / Demucs
    if voice_models_loaded:
        try:
            if 'demucs_model' in globals() and demucs_model is not None:
                for submodel in demucs_model.models:
                    submodel.to("cpu")
        except Exception:
            pass
        for _var in ("hf_model", "hf_processor", "stt_pipe", "demucs_model"):
            if _var in globals():
                globals()[_var] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        voice_models_loaded = False
    # PaddleOCR
    if paddleocr_model is not None:
        del paddleocr_model
        paddleocr_model = None
    try:
        globals()["paddleocr_lang"] = None
        globals()["ocr_help_shown"] = False
    except Exception:
        pass
    # VLM
    if vlm_active:
        try:
            import subprocess, shutil
            if shutil.which("ollama"):
                subprocess.run(["ollama", "stop", OLLAMA_MODEL],
                                timeout=4, check=False)
        except Exception:
            pass
        finally:
            vlm_active = False

    gc.collect()
    update_unload_button_state()

# -----------------------------------------------------
# 1. 載入 sakura 翻譯模型
# -----------------------------------------------------
print("使用加載 sakura GGUF 模型中，請稍候...")
llm = Llama(
    model_path=qwen_gguf_path,
    n_ctx=4096,           # 上下文長度
    temperature=0.6,
    top_p=0.95,
    repeat_penalty=1.1,
    n_gpu_layers=30,      # 看模型大小去測試
    f16_kv=True,          
)
print("GGUF sakura 模型載入完畢。")

JSON_MEMORY_FILE = "short_term_memory.json"  # 存對話記憶的 JSON
MAX_HISTORY_ROUNDS = 5  # 只保留 5 組對話
# 每次腳本啟動時，若檔案存在就刪除
if os.path.exists(JSON_MEMORY_FILE):
    try:
        os.remove(JSON_MEMORY_FILE)
        print(f"已刪除上次紀錄檔 {JSON_MEMORY_FILE}")
    except Exception as e:
        print(f"刪除紀錄檔時出錯: {e}")

def gguf_translate_text_to_chinese(text: str) -> str:
    """
    使用短期記憶 (JSON) + llama-cpp (GGUF sakura)
    """
    # 1) 載入 memory
    memory_data = load_short_term_memory()

    # 2) 新增一輪 user
    memory_data.append({
        "role": "user",
        "content": [
            {"type": "text", "text": text}
        ]
    })

    # 3) 將 memory 轉成 prompt string
    prompt_str = build_prompt_from_memory(memory_data)

    # 4) llama-cpp 呼叫
    res = llm(
        prompt=prompt_str,
        max_tokens=128,
        stop=["<|im_end|>", "</assistant>", "</user>", "</system>"]
    )
    # res: dict => {"choices": [{"text": "..."}], "tokens":...}
    assistant_text = res["choices"][0]["text"].strip()

    # 5) 加到 memory
    memory_data.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": assistant_text}
        ]
    })

    # 若超過對話輪數 => 刪舊
    memory_data = trim_excess_rounds(memory_data)

    # 6) 寫回 JSON
    save_short_term_memory(memory_data)

    return assistant_text

def gguf_translate_text_direct(text: str) -> str:
    """
    不使用記憶，單純呼叫 llama-cpp (GGUF sakura).
    """
    # -------------------------------
    # Prompt 寫法
    # -------------------------------
    prompt_str = (
        "<|im_start|>system\n"
        "你是一個轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
        "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"将下面的日文文本翻译成中文：{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    res = llm(
        prompt=prompt_str,
        max_tokens=128,
        stop=["<|im_end|>", "</assistant>", "</user>", "</system>"]
    )

    assistant_text = res["choices"][0]["text"].strip()
    return assistant_text

def build_prompt_from_memory(memory_data):
    """
    將 memory_data (含 system, user, assistant) 拼成 Qwen2.5 相容的對話 prompt
    使用 <|im_start|>system、<|im_start|>user、<|im_start|>assistant 標記
    """
    # system 部分
    system_text = memory_data[0]["content"][0]["text"]
    prompt = (
        "<|im_start|>system\n"
        f"{system_text}\n"
        "<|im_end|>\n"
    )

    # 從第 1 筆(索引=1)開始，輪流累加 user/assistant
    for item in memory_data[1:]:
        if item["role"] == "user":
            user_txt = "".join(c["text"] for c in item["content"])
            prompt += (
                "<|im_start|>user\n"
                f"{user_txt}\n"
                "<|im_end|>\n"
            )
        else:
            # assistant
            asst_txt = "".join(c["text"] for c in item["content"])
            prompt += (
                "<|im_start|>assistant\n"
                f"{asst_txt}\n"
                "<|im_end|>\n"
            )

    # 最後再開一個 assistant 區塊，讓模型繼續生成
    prompt += "<|im_start|>assistant\n"
    return prompt

def trim_excess_rounds(memory_data):
    MAX_HISTORY_ROUNDS = 5
    system_part = memory_data[0]
    conv_part = memory_data[1:]  # user/assistant pairs
    rounds = len(conv_part)//2
    while rounds > MAX_HISTORY_ROUNDS:
        conv_part = conv_part[2:]  # 砍最舊的一組 (user+assistant)
        rounds = len(conv_part)//2
    return [system_part] + conv_part

# -----------------------------------------------------
# 2. 載入 Hugging Face Whisper large-v3 模型、Demucs 模型
# -----------------------------------------------------
def load_voice_models():
    """
    當用戶選擇語音轉錄模式時，動態載入語音相關模型（Whisper、Demucs）。
    如果已加載則不再重覆加載。
    """
    global hf_model, hf_processor, stt_pipe, demucs_model, torch_device, voice_models_loaded
    if voice_models_loaded:
        return
    print("開始載入語音模型...")
    # 載入 Whisper STT 模型
    device_idx = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_whisper_path,
        torch_dtype=torch_dtype,
        local_files_only=True
    )
    # 強制 decoder 提示清空，避免與 language= 衝突
    hf_model.config.forced_decoder_ids = None
    hf_processor = AutoProcessor.from_pretrained(hf_whisper_path, local_files_only=True)
    stt_pipe = pipeline(
        task="automatic-speech-recognition",
        model=hf_model,
        tokenizer=hf_processor.tokenizer,
        feature_extractor=hf_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device_idx
    )
    print("Whisper 模型載入完成。")
    print("開始載入Demucs模型...")
    # 載入 Demucs 模型
    demucs_model = demucs_get_model(name="htdemucs")
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    for submodel in demucs_model.models:
        submodel.to(torch_device)
    demucs_model.eval()
    print("Demucs 模型載入完成。")
    voice_models_loaded = True
    update_unload_button_state()

# -----------------------------------------------------
# 3. 其餘參數 & GUI
# -----------------------------------------------------

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  #16kHz
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)   # 480
VAD_SENSITIVITY = 1
#RMS_THRESHOLD = 400    # 已改用可選音量閾值
MAX_SILENCE_CHUNKS = 5 # 若連續5個chunk無聲 => 結束錄音

RECORD_SECONDS_LIMIT = 10  # 最長錄幾秒避免拖太長

audio = None
stream = None
virtual_device_name = "CABLE Output"  # 耳機輸出

def open_audio_stream():
    global audio, stream
    if audio is not None and stream is not None:
        return
    audio = pyaudio.PyAudio()
    device_index = None
    for i in range(audio.get_device_count()):
        devinfo = audio.get_device_info_by_index(i)
        if virtual_device_name.lower() in devinfo['name'].lower():
            device_index = i
            break
    if device_index is None:
        print(f"未找到音頻裝置: {virtual_device_name}")
        return  # 不 exit，讓 GUI 還能用 OCR
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=device_index
        )
    except Exception as e:
        print(f"無法打開音頻流: {e}")
        stream = None

def close_audio_stream():
    global audio, stream
    try:
        if stream is not None:
            stream.stop_stream()
            stream.close()
    except Exception:
        pass
    try:
        if audio is not None:
            audio.terminate()
    except Exception:
        pass
    stream = None
    audio = None

# 定義兩個隊列：一個用於錄音線程到處理線程，另一個用於處理線程到GUI
processing_queue = queue.Queue()
gui_queue = queue.Queue()

vad = webrtcvad.Vad(VAD_SENSITIVITY)
transcribe_flag = threading.Event()
#=================================
# GUI
#=================================
root = tk.Tk()
root.title("即時字幕")
root.geometry("600x1020")

gui_text = ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
gui_text.pack(expand=True, fill='both')
gui_text.tag_config("ja_color", foreground="green")
gui_text.tag_config("zh_color", foreground="blue")

#=================================
# 第 1 行：選擇語言來源
#=================================
language_frame = tk.Frame(root)
language_frame.pack(pady=10)

language_label = tk.Label(language_frame, text="選擇來源語言:")
language_label.pack(side=tk.LEFT, padx=5)

language_var = tk.StringVar(value="japanese")  # 預設為日文
language_menu = ttk.Combobox(
    language_frame,
    textvariable=language_var,
    values=["japanese", "english", "chinese"],
    state="readonly",
    width=20
)
language_menu.pack(side=tk.LEFT, padx=5)

#=================================
# 第 2 行：模式選擇 + “使用上下文記憶”
#=================================
mode_context_frame = tk.Frame(root)
mode_context_frame.pack(pady=10)

mode_label = tk.Label(mode_context_frame, text="模式選擇:")
mode_label.pack(side=tk.LEFT, padx=5)

mode_var = tk.StringVar(value=MODE_VOICE)  # 默認 "voice"
mode_menu = ttk.Combobox(
    mode_context_frame,
    textvariable=mode_var,
    values=[MODE_VOICE, MODE_OCR],
    state="readonly",
    width=10
)
mode_menu.pack(side=tk.LEFT, padx=5)

context_var = tk.BooleanVar(value=False)
context_check = tk.Checkbutton(mode_context_frame, text="使用上下文記憶", variable=context_var)
context_check.pack(side=tk.LEFT, padx=5)

#=================================
# 第 3 行：選取區域 + 持續辨識 + 間隔
#=================================
ocr_frame = tk.Frame(root)
ocr_frame.pack(pady=10)

ocr_button = tk.Button(ocr_frame, text="選取區域", command=lambda: select_region(), state="disabled")
ocr_button.pack(side=tk.LEFT, padx=5)

continuous_button = tk.Button(ocr_frame, text="持續辨識", command=lambda: toggle_persistent_ocr(), state="disabled")
continuous_button.pack(side=tk.LEFT, padx=5)

interval_label = tk.Label(ocr_frame, text="間隔(秒):")
interval_label.pack(side=tk.LEFT, padx=5)

interval_var = tk.StringVar(value="5")
interval_menu = ttk.Combobox(
    ocr_frame,
    textvariable=interval_var,
    values=[str(i) for i in range(1, 11)],
    state="readonly",
    width=3
)
interval_menu.pack(side=tk.LEFT, padx=5)

def on_mode_change(event):
    mode = mode_var.get()
    if mode == MODE_OCR:
        ocr_button.config(state="normal")
        continuous_button.config(state="normal")
    else:
        ocr_button.config(state="disabled")
        continuous_button.config(state="disabled")

mode_menu.bind("<<ComboboxSelected>>", on_mode_change)

#=================================
# 第 4 行：卸載 + 音量閾值
#=================================
control_frame = tk.Frame(root)
control_frame.pack(pady=10)
unload_btn = ttk.Button(control_frame,
                        text="手動卸載STT/OCR/VLM",
                        command=unload_models,
                        state="disabled")          # 預設不可按
unload_btn.pack(side="left", padx=4)

threshold_label = tk.Label(control_frame, text="音量閾值:")
threshold_label.pack(side=tk.LEFT, padx=5)

rms_threshold_var = tk.StringVar(value="400")           # 預設 400
threshold_menu = ttk.Combobox(
    control_frame,
    textvariable=rms_threshold_var,
    values=[str(i) for i in range(100, 1001, 100)],     # 100,200,…,1000
    state="readonly",
    width=5
)
threshold_menu.pack(side=tk.LEFT, padx=5)
#=================================
# 第 5 行：文字排版方向
#=================================
orientation_frame = tk.Frame(root)
orientation_frame.pack(pady=10)

orientation_label = tk.Label(orientation_frame, text="上傳圖片的文字方向:")
orientation_label.pack(side=tk.LEFT, padx=5)

orientation_var = tk.StringVar(value="vertical")  # 預設「由上到下再由右到左」

orientation_menu = ttk.Combobox(
    orientation_frame,
    textvariable=orientation_var,
    values=["vertical (由上到下再由右到左)", "horizontal (由左到右再由上到下)"],
    state="readonly",
    width=30
)
orientation_menu.pack(side=tk.LEFT, padx=5)
#=================================
# 第 6 行：VLM模型辨識
#=================================
use_ollama_var = tk.BooleanVar(value=False)
chk_ollama = tk.Checkbutton(root, text="使用ollama運行VLM模型辨識圖片文字", variable=use_ollama_var)
chk_ollama.pack(anchor="w", padx=10, pady=2)

# =========== Demucs 模組 ===========
def compute_rms(data_bytes: bytes) -> float:
    data_i16 = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(data_i16**2)))


def upsample_and_stereo(audio_np: np.ndarray, sr_in=16000, sr_out=44100):
    # 升採樣+stereo
    audio_44k = resampy.resample(audio_np, sr_in, sr_out)
    audio_stereo = np.stack([audio_44k, audio_44k], axis=0)  # shape=(2, samples)
    return audio_stereo, sr_out


def demucs_vocals(audio_bytes: bytes) -> bytes:
    """
    給 16k mono int16 bytes => 升44.1k stereo => demucs => 下采回16k => return int16(16k,mono) bytes
    """
    # => float32
    raw_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # 升採樣+stereo => (2, samples)
    audio_stereo, sr_out = upsample_and_stereo(raw_data, sr_in=RATE, sr_out=44100)
    # demucs
    wave_torch = torch.from_numpy(audio_stereo).unsqueeze(0).to(torch_device)  # (1,2,samples)
    with torch.no_grad():
        separated = demucs_apply_model(demucs_model, wave_torch, overlap=0.25)  
        # shape=(1, n_sources, 2, samples)
        # htdemucs => index=3 => vocals
        vocals_torch = separated[0, 3, :, :]  # shape=(2, samples)

    vocals_np = vocals_torch.cpu().numpy()  # shape=(2, samples)
    # 下采回16k => (samples,) => int16 => bytes
    # 先合成mono => mean axis=0 or axis=?
    vocals_mono = vocals_np.mean(axis=0)  # shape=(samples,)
    down_np = resampy.resample(vocals_mono, sr_out, RATE)
    vocals_i16 = (down_np * 32768.0).clip(-32768,32767).astype(np.int16).tobytes()
    return vocals_i16

def record_audio():
    """
    錄音線程：
    1) 音量>RMS_THRESHOLD => 進入 'recording' 狀態
    2) 在 'recording' 狀態，用 webrtcvad 判斷是否 speech => 有 => 累積, 無 => 結束
    3) 結束後，將錄音數據推送到 processing_queue
    """
    gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 開始錄音(音量閾值+VAD)...\n")

    while not transcribe_flag.is_set():
        try:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except Exception as e:
            gui_queue.put(f"[錄音錯誤] {e}\n")
            continue

        # Step A: 若音量低於閾值 => 不錄
        current_threshold = int(rms_threshold_var.get())
        rms_val = compute_rms(chunk)
        if rms_val < current_threshold:
            continue

        # ========== 一旦音量>=閾值 => 進入'錄音'狀態 ==========
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 音量{rms_val:.0f}>=閾值 {current_threshold},開始錄...\n")

        recorded_frames = []
        silence_count = 0
        # 最長錄 RECORD_SECONDS_LIMIT秒 => 30ms * n
        max_frames = int((RECORD_SECONDS_LIMIT * 1000) / CHUNK_DURATION_MS)
        frame_idx = 0

        # 先把當前chunk納入(因為已經音量>=閾值)
        recorded_frames.append(chunk)
        # 用webrtcvad判斷 => speech就累積, 無speech => silence_count++ => 連續幾次就結束
        while not transcribe_flag.is_set():
            # webrtcvad
            is_speech = vad.is_speech(chunk, RATE)
            if is_speech:
                silence_count = 0
            else:
                silence_count += 1
                if silence_count >=  MAX_SILENCE_CHUNKS:
                    # 連續幾個 chunk 無聲 => 結束
                    break

            frame_idx += 1
            if frame_idx >= max_frames:
                # 避免無限錄
                break

            # 再讀下一個 chunk
            try:
                chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                recorded_frames.append(chunk)
            except Exception as e:
                gui_queue.put(f"[錄音中斷] {e}\n")
                break

        # ========== 錄音結束 => 合併 frames => 推送到 processing_queue ==========
        raw_data_bytes = b''.join(recorded_frames)
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 錄音結束, frames={len(recorded_frames)}\n")

        # 推送到 processing_queue
        processing_queue.put(raw_data_bytes)

    # 結束 => 不需推送 None，因為處理線程會持續等待
    gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 停止錄製(音量閾值+VAD)...\n")

def transcribe_audio():
    """
    處理線程：
    1) 從 processing_queue 獲取錄音數據
    2) Demucs 分離人聲
    3) 計算人聲音量 >200
    4) 若大於200，則用 Whisper 做 STT
    5) 用 Qwen 做翻譯
    6) 將結果推送到 gui_queue
    """
    gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 開始轉錄音頻...\n")
    while not transcribe_flag.is_set() or not processing_queue.empty():
        try:
            raw_data_bytes = processing_queue.get(timeout=1)
        except queue.Empty:
            continue

        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 開始處理錄音...\n")

        try:
            # 1) Demucs分離 => vocals(16k,mono)
            vocals_int16 = demucs_vocals(raw_data_bytes)

            # 2) 計算分離後音量
            post_vocal_threshold = int(rms_threshold_var.get()) * 0.5
            rms_val = compute_rms(vocals_int16)
            if rms_val < post_vocal_threshold:
                gui_queue.put("無人聲跳過\n")
                continue

            # 3) Whisper STT
            audio_np = np.frombuffer(vocals_int16, dtype=np.int16).astype(np.float32) / 32768.0
            audio_input = {
                "array": audio_np,
                "sampling_rate": RATE
            }
            source_language = language_var.get()
            stt_result = stt_pipe(audio_input, generate_kwargs={"language": source_language})
            ja_text = stt_result["text"].strip()

            # 4) 翻譯
            if context_var.get():
                zh_text = gguf_translate_text_to_chinese(ja_text)
            else:
                zh_text = gguf_translate_text_direct(ja_text)

            # 5) 顯示
            gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 轉錄完成。\n")
            gui_queue.put((f"轉錄結果({source_language}): {ja_text}\n", "ja_color"))
            gui_queue.put((f"翻譯結果(中文): {zh_text}\n", "zh_color"))
        except Exception as e:
            gui_queue.put(f"[Demucs/轉錄/翻譯錯誤]: {e}\n")

    gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 停止轉錄音頻。\n")

def select_region():
    """
    創建一個半透明的、可拖拽和可調整大小的選擇窗口，
    用戶調整後點擊“確認”，返回選定區域的截圖文件路徑，
    並調用 PaddleOCR 進行 OCR，再將 OCR 原文和 LLM 翻譯結果顯示到 GUI。
    """
    import pyautogui
    show_ocr_help_once(clear=True)
    class SelectionWindow(tk.Toplevel):
        def __init__(self, master):
            super().__init__(master)
            # 無標題欄
            self.overrideredirect(True)
            # 半透明
            self.attributes("-alpha", 0.3)
            # 初始大小與位置
            self.geometry("300x200+200+200")
            # 設置最小尺寸，避免過度縮小導致消失
            self.minsize(50, 50)

            # ---------- 外層黑色邊框 ----------
            self.outer_frame = tk.Frame(self, bg="black", bd=2, highlightthickness=2, highlightbackground="black")
            self.outer_frame.pack(expand=True, fill="both")

            # ---------- 上半部分：真正要截圖的內容區 ----------
            self.main_frame = tk.Frame(self.outer_frame, bg="gray")
            self.main_frame.pack(side="top", expand=True, fill="both")

            # ---------- 下方：按鈕區域 ----------
            self.button_frame = tk.Frame(self.outer_frame, bg="white")
            self.button_frame.pack(side="bottom", fill="x")

            self.confirm_button = tk.Button(self.button_frame, text="確認", command=self.confirm)
            self.confirm_button.pack()

            # 綁定事件：用戶只在 main_frame 上拖動或縮放
            self.main_frame.bind("<ButtonPress-1>", self.start_move)
            self.main_frame.bind("<B1-Motion>", self.do_move)
            self.main_frame.bind("<ButtonPress-3>", self.start_resize)
            self.main_frame.bind("<B3-Motion>", self.do_resize)

            self.selected_geom = None
            self.start_x = 0
            self.start_y = 0
            self.resize_start_x = 0
            self.resize_start_y = 0
            self.orig_width = 300
            self.orig_height = 200

        def start_move(self, event):
            self.start_x = event.x
            self.start_y = event.y

        def do_move(self, event):
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            new_x = self.winfo_x() + dx
            new_y = self.winfo_y() + dy
            self.geometry(f"+{new_x}+{new_y}")

        def start_resize(self, event):
            self.resize_start_x = event.x
            self.resize_start_y = event.y
            self.orig_width = self.winfo_width()
            self.orig_height = self.winfo_height()

        def do_resize(self, event):
            dx = event.x - self.resize_start_x
            dy = event.y - self.resize_start_y
            new_w = self.orig_width + dx
            new_h = self.orig_height + dy
            if new_w < 50:
                new_w = 50
            if new_h < 50:
                new_h = 50
            self.geometry(f"{new_w}x{new_h}")

        def confirm(self):
            """
            用戶點擊確認後，只計算 main_frame 區域的絕對坐標與大小。
            這樣按鈕區就不會被包含進截圖內。
            """
            self.update_idletasks()
            # 拿到整個Toplevel窗口的geometry
            top_geom = self.geometry()  # e.g. "300x200+200+200"
            size_part, pos_part = top_geom.split("+", 1)
            top_w, top_h = size_part.split("x")
            top_x, top_y = pos_part.split("+")
            top_x, top_y = int(top_x), int(top_y)

            # main_frame 在Toplevel里的相對位置
            main_x = self.main_frame.winfo_x()
            main_y = self.main_frame.winfo_y()
            main_w = self.main_frame.winfo_width()
            main_h = self.main_frame.winfo_height()

            # 把 main_frame 的左上角映射到屏幕絕對坐標
            abs_x = top_x + main_x
            abs_y = top_y + main_y

            # 最終只截取 main_frame 的大小
            self.selected_geom = (abs_x, abs_y, main_w, main_h)
            self.destroy()
          
    sel_win = SelectionWindow(root)
    root.wait_window(sel_win)
    if sel_win.selected_geom is None:
        return  # 用戶未確認
    x, y, width, height = sel_win.selected_geom
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    # OCR處理
    # 根據用戶選擇語言，可動態調整 PaddleOCR 的 lang 參數，例如：若 language_var.get() == "english"，則 lang="en"
    lang_sel = language_var.get()
    if lang_sel == "japanese":
        ocr_lang = "japan"
    elif lang_sel == "english":
        ocr_lang = "en"
    elif lang_sel == "chinese":
        ocr_lang = "ch"
    else:
        ocr_lang = "japan"

    global paddleocr_model, paddleocr_lang
    load_paddleocr(ocr_lang)

    img_np = np.array(screenshot)              # PIL → ndarray(RGB)
    result = paddleocr_model.ocr(img_np, cls=True)
    ocr_text = ""
    for line in result:
        ocr_text += " ".join([w[1][0] for w in line]) + "\n"
    if not ocr_text.strip():
        gui_queue.put("OCR 為空，跳過翻譯\n")
        return
    # 顯示 OCR 原文
    gui_queue.put((f"[{datetime.now().strftime('%H:%M:%S')}] OCR 原文:\n{ocr_text}\n", "ja_color"))
    # 翻譯 OCR 結果（可選擇使用上下文記憶）
    if context_var.get():
        translated = gguf_translate_text_to_chinese(ocr_text)
    else:
        translated = gguf_translate_text_direct(ocr_text)
    gui_queue.put((f"[{datetime.now().strftime('%H:%M:%S')}] 翻譯結果:\n{translated}\n", "zh_color"))    

class PersistentRegionWindow(tk.Toplevel):  # 持續存在的視窗
    def __init__(self, master):
        super().__init__(master)
        # 無標題欄
        self.overrideredirect(True)
        # 半透明
        self.attributes("-alpha", 0.3)
        # 初始大小與位置
        self.geometry("300x200+300+200")
        # 設置最小尺寸
        self.minsize(50, 50)

        # 用 Frame 顯示一個黑色邊框
        self.outer_frame = tk.Frame(self, bg="black", bd=2, highlightthickness=2, highlightbackground="black")
        self.outer_frame.pack(expand=True, fill="both")

        # 上半部分: 真正要截圖的內容區
        self.main_frame = tk.Frame(self.outer_frame, bg="gray")
        self.main_frame.pack(side="top", expand=True, fill="both")

        # 底部: 放提示文字(或按鈕)
        self.bottom_frame = tk.Frame(self.outer_frame, bg="white")
        self.bottom_frame.pack(side="bottom", fill="x")

        self.label_info = tk.Label(self.bottom_frame, text="持續辨識中...")
        self.label_info.pack()

        # 只對 main_frame 綁定拖曳/縮放事件
        self.main_frame.bind("<ButtonPress-1>", self.start_move)
        self.main_frame.bind("<B1-Motion>", self.do_move)
        self.main_frame.bind("<ButtonPress-3>", self.start_resize)
        self.main_frame.bind("<B3-Motion>", self.do_resize)

        # 記錄幾何
        self.selected_geom = None
        self.start_x = 0
        self.start_y = 0
        self.resize_start_x = 0
        self.resize_start_y = 0
        self.orig_width = 300
        self.orig_height = 200

    def start_move(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def do_move(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        new_x = self.winfo_x() + dx
        new_y = self.winfo_y() + dy
        self.geometry(f"+{new_x}+{new_y}")

    def start_resize(self, event):
        self.resize_start_x = event.x
        self.resize_start_y = event.y
        self.orig_width = self.winfo_width()
        self.orig_height = self.winfo_height()

    def do_resize(self, event):
        dx = event.x - self.resize_start_x
        dy = event.y - self.resize_start_y
        new_w = self.orig_width + dx
        new_h = self.orig_height + dy
        if new_w < 50:
            new_w = 50
        if new_h < 50:
            new_h = 50
        self.geometry(f"{new_w}x{new_h}")

    def get_region(self):
        """
        只回傳 main_frame 的絕對座標與大小
        """
        self.update_idletasks()
        # 取得整個 Toplevel 幾何
        top_geom = self.geometry()  # e.g. "300x200+300+200"
        size_part, pos_part = top_geom.split("+", 1)
        top_x, top_y = pos_part.split("+")
        top_x, top_y = int(top_x), int(top_y)

        # 取得 main_frame 在 Toplevel 裡的相對位置與大小
        main_x = self.main_frame.winfo_x()
        main_y = self.main_frame.winfo_y()
        main_w = self.main_frame.winfo_width()
        main_h = self.main_frame.winfo_height()

        abs_x = top_x + main_x
        abs_y = top_y + main_y
        return (abs_x, abs_y, main_w, main_h)
    
def periodic_ocr():
    """
    若 persistent_ocr_active 為 True，則每隔 X 秒自動截取 persistent_window 範圍做OCR
    """
    global persistent_window, persistent_ocr_active
    show_ocr_help_once(clear=True)
    if not persistent_ocr_active or not persistent_window:
        return  # 已停止或窗口不存在
    # 取得當前幾何
    x, y, w, h = persistent_window.get_region()
    import pyautogui
    screenshot = pyautogui.screenshot(region=(x, y, w, h))

    # 動態設置語言
    lang_sel = language_var.get()
    if lang_sel == "japanese":
        ocr_lang = "japan"
    elif lang_sel == "english":
        ocr_lang = "en"
    elif lang_sel == "chinese":
        ocr_lang = "ch"
    else:
        ocr_lang = "japan"

    load_paddleocr(ocr_lang)
    img_np = np.array(screenshot)
    result = paddleocr_model.ocr(img_np, cls=True)
    ocr_text = ""
    for line in result:
        ocr_text += " ".join([w[1][0] for w in line]) + "\n"
    
    # 如果無內容 => 直接跳過
    if not ocr_text.strip():
        # 再次調度
        try:
            interval_sec = int(interval_var.get())
        except:
            interval_sec = 5
        if persistent_ocr_active:
            root.after(interval_sec * 1000, periodic_ocr)
        return

    gui_queue.put((f"[{datetime.now().strftime('%H:%M:%S')}] OCR(持續)原文:\n{ocr_text}\n", "ja_color"))

    # 翻譯
    if context_var.get():
        translated = gguf_translate_text_to_chinese(ocr_text)
    else:
        translated = gguf_translate_text_direct(ocr_text)
    gui_queue.put((f"[{datetime.now().strftime('%H:%M:%S')}] 翻譯結果:\n{translated}\n", "zh_color"))

    # 讀取下拉選單間隔
    try:
        interval_sec = int(interval_var.get())
    except:
        interval_sec = 5
    if persistent_ocr_active:
        root.after(interval_sec * 1000, periodic_ocr)

def toggle_persistent_ocr():
    """
    如果目前未啟動 => 創建窗口 + 倒數5秒後開始 OCR；
    如果已啟動 => 停止並銷毀窗口。
    """
    global persistent_window, persistent_ocr_active
    if persistent_ocr_active:
        # 當前在持續辨識 => 關閉
        persistent_ocr_active = False
        if persistent_window:
            persistent_window.destroy()
            persistent_window = None
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 持續辨識已停止\n")
    else:
        # 未啟動 => 創建窗口
        persistent_ocr_active = True
        persistent_window = PersistentRegionWindow(root)
        show_ocr_help_once(clear=True)
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 5秒後開始持續辨識...\n")
        # 5秒後開始
        root.after(5000, start_persistent_ocr)

def start_persistent_ocr():
    """
    5秒等待結束後，開始真正的 periodic_ocr
    """
    if not persistent_ocr_active or not persistent_window:
        return  
    gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 持續辨識啟動\n")
    periodic_ocr()  # 第一次立即執行
        
def update_gui():
    """
    定期檢查 gui_queue，並更新 GUI
    """
    while not gui_queue.empty():
        try:
            msg = gui_queue.get_nowait()

            if msg == "__CLEAR__":
                gui_text.delete(1.0, tk.END)
                continue
            if isinstance(msg, tuple) and len(msg) == 2:
                # 用於標色顯示
                text, tag = msg
                gui_text.insert(tk.END, text, tag)
            else:
                gui_text.insert(tk.END, msg)
            gui_text.see(tk.END)  # 確保每次插入後都滾動
        except queue.Empty:
            pass
    update_unload_button_state() 
    root.after(100, update_gui)  # 每100ms檢查一次

def show_ocr_help_once(clear=True):
    """第一次顯示 OCR 說明；之後不再顯示。"""
    global ocr_help_shown
    if not ocr_help_shown:
        if clear:
            gui_queue.put("__CLEAR__")  # 先清空 GUI
        gui_queue.put(
            f"[{datetime.now().strftime('%H:%M:%S')}] OCR 模式啟動\n"
            "請點擊 '選取區域' 按鈕進行單次辨識\n"
            "'持續辨識' 按鈕進持續辨識\n"
            "右鍵拖拽可改區域大小\n"
            "左鍵拖拽可移動\n"
        )
        ocr_help_shown = True

def start_transcription():
    global record_thread, transcribe_thread, voice_models_loaded
    transcribe_flag.clear()
    gui_text.delete(1.0, tk.END)
    current_mode = mode_var.get()
    if current_mode == MODE_VOICE:
        # 僅語音模式才開音訊流
        open_audio_stream()
        if stream is None:
            gui_queue.put("音訊裝置開啟失敗，請確認裝置或名稱。\n")
            return
        # 動態加載語音相關模型（Whisper、Demucs）只在首次選擇時加載
        if not voice_models_loaded:
            load_voice_models()  # 載入
        record_thread = threading.Thread(target=record_audio, daemon=True)
        transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)
        record_thread.start()
        transcribe_thread.start()
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 語音轉錄模式啟動。\n")
    else:
        # 根據下拉選單決定 OCR 語言
        lang_sel = language_var.get()
        if lang_sel == "japanese":
            ocr_lang = "japan"
        elif lang_sel == "english":
            ocr_lang = "en"
        elif lang_sel == "chinese":
            ocr_lang = "ch"
        else:
            ocr_lang = "japan"

        load_paddleocr(ocr_lang)
        global ocr_help_shown
        if not ocr_help_shown:
            gui_queue.put(
                f"[{datetime.now().strftime('%H:%M:%S')}] OCR 模式啟動\n"
                "請點擊 '選取區域' 按鈕進行單次辨識\n"
                "'持續辨識' 按鈕進持續辨識\n"
                "右鍵拖拽可改區域大小\n"
                "左鍵拖拽可移動\n"
            )
            ocr_help_shown = True
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

# -------------- 辨識圖片小工具 --------------
def put_log(msg, tag=None):
    try:
        if tag:
            gui_queue.put((msg, tag))
        else:
            gui_queue.put(msg)
    except Exception:
        pass

def get_font(px):
    try:
        return ImageFont.truetype("NotoSansCJK-Bold.otf", px)
    except Exception:
        try:
            return ImageFont.truetype("msyh.ttc", px)
        except Exception:
            return ImageFont.load_default()

def draw_text_with_stroke(draw, xy, text, font,
                            fill=(255, 255, 255),
                            stroke_fill=(0, 0, 0),
                            stroke_width=2):
    x, y = xy
    for dx in (-stroke_width, 0, stroke_width):
        for dy in (-stroke_width, 0, stroke_width):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=stroke_fill)
    draw.text((x, y), text, font=font, fill=fill)

def measure_line(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2], bb[3]

# === OCR 強化：先放大 → 多路增強（含反相）→ 分別跑 PaddleOCR → 擇優 ===
def _resize_long_side(img_rgb, target_long=1536):
    h, w = img_rgb.shape[:2]
    long_side = max(h, w)
    if long_side >= target_long:
        return img_rgb
    scale = target_long / float(long_side)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def _prep_variants(img_rgb):
    base = img_rgb

    # v0: 原圖
    v0 = base

    # v1: CLAHE on L channel
    lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    v1 = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

    # v2: 自適應二值（白底黑字）
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    v2g = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 10)
    v2 = cv2.cvtColor(v2g, cv2.COLOR_GRAY2RGB)

    # v3: 自適應二值 + 反相（黑底白字）
    v3g = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 10)
    v3 = cv2.cvtColor(v3g, cv2.COLOR_GRAY2RGB)

    # v4: 輕度銳化
    blur = cv2.GaussianBlur(base, (0, 0), 1.0)
    v4 = cv2.addWeighted(base, 1.5, blur, -0.5, 0)

    # v5: 雙邊去噪 + CLAHE
    bf = cv2.bilateralFilter(base, d=5, sigmaColor=50, sigmaSpace=50)
    lab2 = cv2.cvtColor(bf, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab2)
    l3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    v5 = cv2.cvtColor(cv2.merge([l3, a, b]), cv2.COLOR_LAB2RGB)

    # v6: 向左旋轉90度
    v6 = cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return [v0, v1, v2, v3, v4, v5, v6]

def ocr_best_of_upscaled(img_rgb):
    up = _resize_long_side(img_rgb, target_long=1536)
    variants = _prep_variants(up)
    best = (None, "", -1.0, -1)  # (lines, text, score, idx)

    def _score_one(img_arr):
        try:
            res = paddleocr_model.ocr(img_arr, cls=True)
        except Exception:
            return None, "", 0.0
        lines = res[0] if (res and isinstance(res[0], list)) else (res if isinstance(res, list) else [])
        texts, scores = [], []
        for ln in lines:
            try:
                t, s = ln[1][0], float(ln[1][1])
            except Exception:
                t, s = "", 0.0
            if t:
                texts.append(t)
                scores.append(s)
        if not texts:
            return None, "", 0.0
        avg = sum(scores) / len(scores)
        return lines, " ".join(texts), avg

    # 對 v1~v5 的製作（供 v6 二次處理使用）
    def _prep_v1_v5_only(base):
        out = []
        # v1
        lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        out.append(cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB))
        # v2
        gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        v2g = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, 10)
        out.append(cv2.cvtColor(v2g, cv2.COLOR_GRAY2RGB))
        # v3
        v3g = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 35, 10)
        out.append(cv2.cvtColor(v3g, cv2.COLOR_GRAY2RGB))
        # v4
        blur = cv2.GaussianBlur(base, (0, 0), 1.0)
        out.append(cv2.addWeighted(base, 1.5, blur, -0.5, 0))
        # v5
        bf = cv2.bilateralFilter(base, d=5, sigmaColor=50, sigmaSpace=50)
        lab2 = cv2.cvtColor(bf, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab2)
        l3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out.append(cv2.cvtColor(cv2.merge([l3, a, b]), cv2.COLOR_LAB2RGB))
        return out

    mode_str = orientation_var.get()
    is_vertical = mode_str.startswith("vertical")
    is_horizontal = mode_str.startswith("horizontal")

    for idx, v in enumerate(variants):
        lines, text, avg = _score_one(v)
        if not lines:
            continue
        # 只針對 v6 做加/降權與二次處理
        if idx == 6:
            if is_vertical:
                # 縱排 → 先 *1.5
                avg *= 1.5

                # 若加權後仍然 < 1.12 → 對 v6 的旋轉圖再做一次 v1~v5 增強與 OCR
                if avg < 1.12:
                    print("[OCR] v6 加權後仍 < 1.12，對旋轉後的圖再做 v1~v5 增強後重試")
                    candidates = _prep_v1_v5_only(v)  # v 是 v6 圖（已旋轉）

                    best2 = (None, "", 0.0)
                    for vv in candidates:
                        l2, t2, a2 = _score_one(vv)
                        if a2 > best2[2]:
                            best2 = (l2, t2, a2)

                    # 若二次處理有結果，將其作為 v6 的最終結果（同樣套用 *1.5）
                    if best2[0] is not None:
                        lines, text, avg = best2[0], best2[1], best2[2] * 1.5
                        print(f"[OCR] v6 二次處理生效，新的平均置信度(已加權)={avg:.3f}")
                    else:
                        print("[OCR] v6 二次處理沒有有效結果，沿用原本分數")

            elif is_horizontal:
                # 橫排 → 對 v6 降權
                avg *= 0.7

        # 更新全域最佳
        if avg > best[2]:
            best = (lines, text, avg, idx)

    if best[3] >= 0:
        print(f"[OCR] 最佳版本 v{best[3]} 平均置信度={best[2]:.3f}")
    else:
        print("[OCR] 沒有有效結果")

    return best  # (best_lines, best_text, best_avg_score, best_index)

# === 橫排排版（\n 先分段） ===
def layout_horizontal(draw, text, box_w, box_h, min_font=10):
    segments = text.split("\n") if "\n" in text else [text]
    font_size = max(14, int(box_h * 0.72))
    while font_size >= min_font:
        font = get_font(font_size)
        lines = []
        for seg in segments:
            cur = ""
            for ch in seg:
                w, _ = measure_line(draw, cur + ch, font)
                if w <= max(1, box_w):
                    cur += ch
                else:
                    if cur:
                        lines.append(cur)
                    cur = ch
            if cur:
                lines.append(cur)
        line_h = max(1, measure_line(draw, "字", font)[1])
        total_h = line_h * len(lines)
        if total_h <= box_h or font_size == min_font:
            return font, lines, line_h
        font_size -= 1
    font = get_font(min_font)
    line_h = max(1, measure_line(draw, "字", font)[1])
    return font, [text], line_h

# === 直排排版（\n 當作換欄） ===
def layout_vertical(draw, text, box_w, box_h, min_font=10):
    segs = text.split("\n") if "\n" in text else [text]
    font_size = max(14, int(box_w * 0.72))
    while font_size >= min_font:
        font = get_font(font_size)
        col_w = max(1, measure_line(draw, "字", font)[0])
        step_h = max(1, measure_line(draw, "字", font)[1])
        max_per_col = max(1, box_h // step_h)

        cols = []
        for seg in segs:
            i = 0
            while i < len(seg):
                cols.append(list(seg[i:i + max_per_col]))
                i += max_per_col

        cols.reverse()  # 改為右到左(把欄順序反轉)
        total_w = col_w * max(1, len(cols))
        if total_w <= box_w or font_size == min_font:
            return font, cols, step_h, col_w
        font_size -= 1

    font = get_font(min_font)
    col_w = max(1, measure_line(draw, "字", font)[0])
    step_h = max(1, measure_line(draw, "字", font)[1])
    return font, [list("".join(segs))], step_h, col_w

def render_text_on_full(draw_full, box, text, vertical_mode):
    """
    最終回填到原圖。
    """
    x1, y1, x2, y2 = box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    draw_full.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

    if vertical_mode:
        dummy = Image.new("RGB", (box_w, box_h), (0, 0, 0))
        d = ImageDraw.Draw(dummy)
        font, cols, step_h, col_w = layout_vertical(d, text, box_w, box_h)
        total_w = col_w * len(cols)
        total_h = step_h * (max(len(c) for c in cols) if cols else 1)
        x0 = x1 + max(0, (box_w - total_w) // 2)
        y0 = y1 + max(0, (box_h - total_h) // 2)
        for ci, col in enumerate(cols):
            cx = x0 + ci * col_w
            cy = y0
            for ch in col:
                draw_text_with_stroke(draw_full, (cx, cy), ch, font,
                                        fill=(255, 255, 255), stroke_fill=(0, 0, 0),
                                        stroke_width=max(2, font.size // 12))
                cy += step_h
    else:
        dummy = Image.new("RGB", (box_w, box_h), (0, 0, 0))
        d = ImageDraw.Draw(dummy)
        font, lines, line_h = layout_horizontal(d, text, box_w, box_h)
        total_h = line_h * len(lines)
        y0 = y1 + max(0, (box_h - total_h) // 2)
        for i, ln in enumerate(lines):
            w, _ = measure_line(d, ln, font)
            x = x1 + max(0, (box_w - w) // 2)
            y = y0 + i * line_h
            draw_text_with_stroke(draw_full, (x, y), ln, font,
                                    fill=(255, 255, 255), stroke_fill=(0, 0, 0),
                                    stroke_width=max(2, font.size // 12))

def infer_vertical_from_ocr_lines(lines):
    """用 OCR 行框的寬高投票推斷方向；平手預設橫排(False)。"""
    horiz = 0
    vert = 0
    for ln in lines:
        try:
            poly = np.array(ln[0], dtype=np.float32)
            w = (np.linalg.norm(poly[0]-poly[1]) + np.linalg.norm(poly[2]-poly[3])) / 2.0
            h = (np.linalg.norm(poly[1]-poly[2]) + np.linalg.norm(poly[3]-poly[0])) / 2.0
            if w >= h * 1.1:
                horiz += 1
            elif h >= w * 1.1:
                vert += 1
        except Exception:
            continue
    return vert > horiz

def build_text_mask_from_crop_auto(crop_rgb, dilate_px=3):
    """
    從整個裁切區域自動估計文字遮罩。
    適合漫畫黑描邊白底/灰網點：自適應二值 + 邊緣 → 關閉/膨脹，並移除過大的連通元件避免整塊白框被抹掉。
    回傳 0/255 單通道遮罩（白=要去除的筆劃）。
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    # 自適應二值（反相：文字→白）
    th_adpt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        31, 15
    )
    # 邊緣
    edges = cv2.Canny(gray, 60, 150)

    # 合併兩種線索
    mask = cv2.bitwise_or(th_adpt, edges)

    # 收斂/膨脹，讓字描邊被完整覆蓋
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    # 移除過大的 blob（避免把整個對話框邊框或大片白底一併抹掉）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    h, w = gray.shape[:2]
    too_big = (h * w) * 0.6
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > too_big:
            mask[labels == i] = 0
    return mask

def inpaint_crop_with_mask(crop_rgb, mask, radius=3, method="telea"):
    """
    針對整個裁切區域做 inpaint（OpenCV），用遮罩把筆劃抹掉並補背景。
    """
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    flag = cv2.INPAINT_TELEA if method.lower() != "ns" else cv2.INPAINT_NS
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    out = cv2.inpaint(bgr, mask_u8, float(radius), flag)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def paste_inpaint_then_text_using_box(img_pil_full, box, text, vertical_mode):
    """
    先針對「使用者框選的整塊 box」自動產生遮罩並 inpaint，
    再把譯文畫上去。
    """
    x1, y1, x2, y2 = box
    crop_np = np.array(img_pil_full)[y1:y2, x1:x2, :]
    if crop_np.size == 0:
        return
    # 自動產生遮罩並 inpaint
    mask = build_text_mask_from_crop_auto(crop_np, dilate_px=3)
    crop_inpaint = inpaint_crop_with_mask(crop_np, mask, radius=3, method="telea")

    # 先把 inpaint 好的底圖貼回去
    img_pil_full.paste(Image.fromarray(crop_inpaint), (x1, y1))

    # 接著在貼回後的底圖上做原本的直/橫排文字排版（沿用你現有的版型）
    draw_full = ImageDraw.Draw(img_pil_full)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    if vertical_mode:
        dummy = Image.new("RGB", (box_w, box_h), (0, 0, 0))
        d = ImageDraw.Draw(dummy)
        font, cols, step_h, col_w = layout_vertical(d, text, box_w, box_h)
        total_w = col_w * len(cols)
        total_h = step_h * (max(len(c) for c in cols) if cols else 1)
        x0 = x1 + max(0, (box_w - total_w) // 2)
        y0 = y1 + max(0, (box_h - total_h) // 2)
        for ci, col in enumerate(cols):
            cx = x0 + ci * col_w
            cy = y0
            for ch in col:
                draw_text_with_stroke(draw_full, (cx, cy), ch, font,
                                        fill=(255, 255, 255), stroke_fill=(0, 0, 0),
                                        stroke_width=max(2, font.size // 12))
                cy += step_h
    else:
        dummy = Image.new("RGB", (box_w, box_h), (0, 0, 0))
        d = ImageDraw.Draw(dummy)
        font, lines2, line_h = layout_horizontal(d, text, box_w, box_h)
        total_h = line_h * len(lines2)
        y0 = y1 + max(0, (box_h - total_h) // 2)
        for i, ln in enumerate(lines2):
            w2, _ = measure_line(d, ln, font)
            x = x1 + max(0, (box_w - w2) // 2)
            y = y0 + i * line_h
            draw_text_with_stroke(draw_full, (x, y), ln, font,
                                    fill=(255, 255, 255), stroke_fill=(0, 0, 0),
                                    stroke_width=max(2, font.size // 12))

def render_preview_on_inpainted_crop(crop_inpaint_rgb, text, vertical_mode):
    """
    crop_inpaint_rgb: 已 inpaint 的裁切區域 (H,W,3) RGB
    text: 需顯示的翻譯文字
    vertical_mode: 是否直排
    回傳：BGR numpy，用於 cv2.imshow
    """
    crop_pil = Image.fromarray(crop_inpaint_rgb)  # 直接拿 inpaint 後底圖
    draw_crop = ImageDraw.Draw(crop_pil)
    box_w, box_h = crop_pil.size

    if vertical_mode:
        font, cols, step_h, col_w = layout_vertical(draw_crop, text, box_w, box_h)
        total_w = col_w * len(cols)
        total_h = step_h * (max(len(c) for c in cols) if cols else 1)
        x0 = max(0, (box_w - total_w) // 2)
        y0 = max(0, (box_h - total_h) // 2)
        for ci, col in enumerate(cols):
            cx = x0 + ci * col_w
            cy = y0
            for ch in col:
                draw_text_with_stroke(draw_crop, (cx, cy), ch, font,
                                        fill=(255, 255, 255),
                                        stroke_fill=(0, 0, 0),
                                        stroke_width=max(2, font.size // 12))
                cy += step_h
    else:
        font, lines, line_h = layout_horizontal(draw_crop, text, box_w, box_h)
        total_h = line_h * len(lines)
        y0 = max(0, (box_h - total_h) // 2)
        for i, ln in enumerate(lines):
            w, _ = measure_line(draw_crop, ln, font)
            x = max(0, (box_w - w) // 2)
            y = y0 + i * line_h
            draw_text_with_stroke(draw_crop, (x, y), ln, font,
                                    fill=(255, 255, 255),
                                    stroke_fill=(0, 0, 0),
                                    stroke_width=max(2, font.size // 12))

    # 顯示前做尺寸限制，避免視窗過大
    prev_bgr = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
    ph, pw = prev_bgr.shape[:2]
    pmax = max(ph, pw)
    if pmax > 720:
        s = pmax / 720.0
        prev_bgr = cv2.resize(prev_bgr, (int(pw / s), int(ph / s)), interpolation=cv2.INTER_AREA)
    return prev_bgr

def upload_and_translate_image():
    """
      - 點按鈕：清空 GUI 並顯示一次操作說明
      - 使用者可一次選多張圖；對每張圖依序執行：
          1) 顯示圖(最長邊≤1080) → 多區域框選（Enter 完成 / Esc 取消該張）
          2) 逐框 OCR+翻譯 → 預覽（V=直排/H=橫排；Y=回填 / N/Esc=跳過）
          3) 回填到原圖
          4) 若有至少一個 Y 才存到 SAVE_DIR/<檔名>_translated.png；若全是 N 則不存
      - 結束一張後自動進入下一張；全部做完即結束
    """
    # —— 主執行緒：清空 GUI 並顯示一次說明（本次多圖批次只顯示一次）——
    try:
        gui_queue.put("__CLEAR__")
        gui_queue.put((
            "【操作說明】在框選視窗用滑鼠拖曳可多選；Enter 完成框選、Esc 取消該張。\n"
            "每個預覽窗：Y=回填、N/Esc=跳過、V=直排、H=橫排。\n"
            "可一次選多張圖片，會依序處理。\n",
            "zh_color"
        ))
    except Exception:
        pass

    def _worker():
        # -------------- 1) 選檔（多選） --------------
        filepaths = filedialog.askopenfilenames(
            title="選擇要翻譯的圖片（可多選）",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        if not filepaths:
            return

        # -------------- 2) 初始化 OCR/翻譯（一次） --------------
        lang_sel = language_var.get() if 'language_var' in globals() else "japanese"
        if lang_sel == "japanese":
            ocr_lang = "japan"
        elif lang_sel == "english":
            ocr_lang = "en"
        elif lang_sel == "chinese":
            ocr_lang = "ch"
        else:
            ocr_lang = "japan"
        load_paddleocr(ocr_lang)
        use_context = context_var.get() if 'context_var' in globals() else False

        # -------------- 3) 逐張圖片處理 --------------
        for idx_img, filepath in enumerate(filepaths, 1):
            try:
                # 用 PIL 讀，避免 cv2 在中文路徑失敗
                img_pil_full = Image.open(filepath).convert("RGB")
            except Exception as e:
                put_log(f"[{idx_img}/{len(filepaths)}] 讀取失敗：{filepath}\n")
                continue

            orig_w, orig_h = img_pil_full.size
            draw_full = ImageDraw.Draw(img_pil_full)

            # 顯示圖 ≤1080
            img_cv_disp = cv2.cvtColor(np.array(img_pil_full), cv2.COLOR_RGB2BGR)
            disp_h, disp_w = img_cv_disp.shape[:2]
            max_side = max(disp_h, disp_w)
            if max_side > 1080:
                scale = max_side / 1080.0
                new_w = int(disp_w / scale)
                new_h = int(disp_h / scale)
                img_cv_disp = cv2.resize(img_cv_disp, (new_w, new_h), interpolation=cv2.INTER_AREA)
                put_log(f"[{idx_img}/{len(filepaths)}] 圖片已縮放到 {new_w}x{new_h} 顯示。\n")
            else:
                new_w, new_h = disp_w, disp_h
            sx = orig_w / float(new_w)
            sy = orig_h / float(new_h)

            base_disp = img_cv_disp.copy()
            cur_disp = img_cv_disp.copy()

            # 多框選
            selecting = False
            start_pt = None
            boxes_disp = []

            win_select = f"選取文字區域 - {os.path.basename(filepath)} (Enter 完成 / Esc 取消)"
            cv2.namedWindow(win_select)

            def mouse_cb(event, x, y, flags, param):
                nonlocal selecting, start_pt, cur_disp, base_disp
                if event == cv2.EVENT_LBUTTONDOWN:
                    selecting = True
                    start_pt = (x, y)
                elif event == cv2.EVENT_MOUSEMOVE and selecting:
                    cur_disp = base_disp.copy()
                    cv2.rectangle(cur_disp, start_pt, (x, y), (0, 255, 0), 2)
                elif event == cv2.EVENT_LBUTTONUP:
                    selecting = False
                    x1, y1 = start_pt
                    x2, y2 = (x, y)
                    xd1, xd2 = sorted([x1, x2])
                    yd1, yd2 = sorted([y1, y2])
                    xd1 = max(0, min(xd1, new_w - 1)); xd2 = max(0, min(xd2, new_w - 1))
                    yd1 = max(0, min(yd1, new_h - 1)); yd2 = max(0, min(yd2, new_h - 1))
                    if xd2 - xd1 >= 5 and yd2 - yd1 >= 5:
                        boxes_disp.append((xd1, yd1, xd2, yd2))
                        cv2.rectangle(base_disp, (xd1, yd1), (xd2, yd2), (0, 0, 255), 2)
                        cur_disp = base_disp.copy()
                        put_log(f"[{idx_img}/{len(filepaths)}] 已選取區域(顯示座標): {xd1},{yd1},{xd2},{yd2}\n")

            cv2.setMouseCallback(win_select, mouse_cb)

            # 非阻塞輪詢
            cancel_this_image = False
            while True:
                cv2.imshow(win_select, cur_disp)
                key = cv2.waitKey(20) & 0xFF
                if key == 13:   # Enter 完成
                    break
                elif key == 27: # Esc 取消該張
                    cancel_this_image = True
                    break
            try:
                cv2.destroyWindow(win_select)
            except Exception:
                pass

            if cancel_this_image:
                put_log(f"[{idx_img}/{len(filepaths)}] 已取消此圖片。\n")
                # 安全關閉預覽殘窗
                try: cv2.destroyAllWindows()
                except: pass
                continue

            if not boxes_disp:
                put_log(f"[{idx_img}/{len(filepaths)}] 未選取任何區域，略過此圖片。\n")
                try: cv2.destroyAllWindows()
                except: pass
                continue

            # 映射至原圖座標
            boxes_orig = []
            for xd1, yd1, xd2, yd2 in boxes_disp:
                xo1 = int(round(xd1 * sx)); yo1 = int(round(yd1 * sy))
                xo2 = int(round(xd2 * sx)); yo2 = int(round(yd2 * sy))
                xo1, xo2 = sorted([max(0, min(xo1, orig_w - 1)), max(0, min(xo2, orig_w - 1))])
                yo1, yo2 = sorted([max(0, min(yo1, orig_h - 1)), max(0, min(yo2, orig_h - 1))])
                if xo2 - xo1 >= 5 and yo2 - yo1 >= 5:
                    boxes_orig.append((xo1, yo1, xo2, yo2))
                    put_log(f"[{idx_img}/{len(filepaths)}] 對應原圖區域: {xo1},{yo1},{xo2},{yo2}\n")

            if not boxes_orig:
                put_log(f"[{idx_img}/{len(filepaths)}] 映射後無有效區域，略過此圖片。\n")
                try: cv2.destroyAllWindows()
                except: pass
                continue

            # 逐框 OCR/翻譯/預覽 → 收集回填
            to_paste = []  # (x1,y1,x2,y2, zh, vertical_mode)
            PREVIEW_WIN = f"預覽 - {os.path.basename(filepath)}"

            for i, (x1, y1, x2, y2) in enumerate(boxes_orig, 1):
                crop_np = np.array(img_pil_full)[y1:y2, x1:x2, :]
                if crop_np.size == 0:
                    put_log(f"[{idx_img}/{len(filepaths)}][區域 {i}] 裁切為空，跳過。\n")
                    continue

                # OCR（放大+多路增強擇優）
                lines, text_detected, avg_score, best_idx = ocr_best_of_upscaled(crop_np)
                if not text_detected.strip():
                    put_log(f"[{idx_img}/{len(filepaths)}][區域 {i}] OCR 無內容，跳過。\n")
                    continue
                put_log(f"[{idx_img}/{len(filepaths)}][區域 {i}] OCR 置信平均分數: {avg_score:.3f}\n")

                # 翻譯
                try:
                    zh = gguf_translate_text_to_chinese(text_detected) if use_context else gguf_translate_text_direct(text_detected)
                except Exception:
                    zh = text_detected
                put_log(f"[{idx_img}/{len(filepaths)}][區域 {i}] {text_detected} => {zh}\n", "zh_color")

                # 初始方向
                try:
                    vertical_mode = orientation_var.get().startswith("vertical")
                except Exception:
                    # 極端情況才退回自動判斷
                    vertical_mode = infer_vertical_from_ocr_lines(lines)

                _auto_mask = build_text_mask_from_crop_auto(crop_np, dilate_px=3)
                _crop_inpaint = inpaint_crop_with_mask(crop_np, _auto_mask, radius=3, method="telea")

                # 非阻塞預覽迴圈（允許 V/H 切換）
                while True:
                    prev_bgr = render_preview_on_inpainted_crop(_crop_inpaint, zh, vertical_mode)
                    cv2.imshow(PREVIEW_WIN, prev_bgr)
                    k = cv2.waitKey(20) & 0xFF
                    if k in (ord('y'), ord('Y')):
                        to_paste.append((x1, y1, x2, y2, zh, vertical_mode))
                        try:
                            cv2.destroyWindow(PREVIEW_WIN)
                        except Exception:
                            pass
                        put_log(f"[{idx_img}/{len(filepaths)}][區域 {i}] → 已加入回填清單。\n")
                        break
                    elif k in (ord('n'), ord('N'), 27):  # N 或 Esc
                        try:
                            cv2.destroyWindow(PREVIEW_WIN)
                        except Exception:
                            pass
                        put_log(f"[{idx_img}/{len(filepaths)}][區域 {i}] → 跳過此框。\n")
                        break
                    elif k in (ord('v'), ord('V')):
                        vertical_mode = True
                    elif k in (ord('h'), ord('H')):
                        vertical_mode = False
                    # 其他鍵：忽略，持續輪詢

            # 統一回填（to_paste 為空就不回填）
            for (x1, y1, x2, y2, zh, vertical_mode) in to_paste:
                paste_inpaint_then_text_using_box(img_pil_full, (x1, y1, x2, y2), zh, vertical_mode)

            # 存檔（只有在有至少一個 Y 時才存）
            base = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(SAVE_DIR, f"{base}_translated.png")
            if to_paste:
                try:
                    img_pil_full.save(out_path)
                    put_log(f"[{idx_img}/{len(filepaths)}] 【完成】已輸出：{out_path}\n", "zh_color")
                except Exception as e:
                    put_log(f"[{idx_img}/{len(filepaths)}] 存檔失敗：{out_path}\n")
            else:
                put_log(f"[{idx_img}/{len(filepaths)}] 未選擇任何框回填，未輸出圖片。\n", "zh_color")

            # 清理殘視窗（保險）
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        # 全部處理完
        put_log("【批次完成】所有選取圖片皆已處理。\n", "zh_color")

    # 啟動背景執行緒（daemon=True 讓程式結束時不阻塞）
    threading.Thread(target=_worker, daemon=True).start()

def _bgr_image_to_png_b64(img_bgr) -> str:
    """把 BGR 的 np.ndarray 轉成 PNG base64。"""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def vlm_recognize_text_from_bgr(crop_bgr) -> str:
    """
    直接把裁切的 BGR 影像陣列送進 VLM（Ollama），回傳模型輸出的文字。
    """
    b64 = _bgr_image_to_png_b64(crop_bgr)
    if not b64:
        return ""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{
            "role": "user",
            "content": "辨識這張圖中的文字並提取出來，只要回覆文字給我，不要解釋或翻譯。",
            "images": [b64],
        }],
        "stream": False,
        "options": {
            "num_gpu": 15      # 看模型大小去測試 3b:35 7b:15
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp = r.json()
        msg = resp.get("message", {})
        content = (msg.get("content") or "").strip()
        return content
    except Exception:
        return ""

def upload_and_translate_image_vlm():
    """與 upload_and_translate_image 幾乎相同，但 OCR 改為 VLM，且不做增強打分。"""
    try:
        gui_queue.put("__CLEAR__")
        gui_queue.put((
            "【操作說明】在框選視窗用滑鼠拖曳可多選；Enter 完成框選、Esc 取消該張。\n"
            "每個預覽窗：Y=回填、N/Esc=跳過、V=直排、H=橫排。\n"
            "可一次選多張圖片，會依序處理。\n",
            "zh_color"
        ))
    except Exception:
        pass
    global vlm_active
    vlm_active = True
    try:
        update_unload_button_state()  
    except Exception:
        pass
    def _worker():
        filepaths = filedialog.askopenfilenames(
            title="選擇要翻譯的圖片（可多選）",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        if not filepaths:
            return

        use_context = context_var.get() if 'context_var' in globals() else False

        for idx_img, filepath in enumerate(filepaths, 1):
            try:
                img_pil_full = Image.open(filepath).convert("RGB")
                img_np_full = np.array(img_pil_full)[:, :, ::-1]  # to BGR
                h, w = img_np_full.shape[:2]
                scale = 1080.0 / max(h, w) if max(h, w) > 1080 else 1.0
                disp_bgr = cv2.resize(img_np_full, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

                boxes_disp = []
                selecting = False
                start_pt = (0, 0)
                base_disp = disp_bgr.copy()   # 保存所有固定的紅框
                cur_disp = base_disp.copy()   # 隨滑鼠移動的畫面

                def mouse_cb(event, x, y, flags, param):
                    nonlocal selecting, start_pt, cur_disp, base_disp
                    if event == cv2.EVENT_LBUTTONDOWN:
                        selecting = True
                        start_pt = (x, y)
                    elif event == cv2.EVENT_MOUSEMOVE and selecting:
                        cur_disp = base_disp.copy()
                        cv2.rectangle(cur_disp, start_pt, (x, y), (0, 255, 0), 2)
                    elif event == cv2.EVENT_LBUTTONUP:
                        selecting = False
                        x1, y1 = start_pt
                        x2, y2 = (x, y)
                        xd1, xd2 = sorted([x1, x2])
                        yd1, yd2 = sorted([y1, y2])
                        xd1 = max(0, min(xd1, disp_bgr.shape[1]-1))
                        xd2 = max(0, min(xd2, disp_bgr.shape[1]-1))
                        yd1 = max(0, min(yd1, disp_bgr.shape[0]-1))
                        yd2 = max(0, min(yd2, disp_bgr.shape[0]-1))
                        if xd2 - xd1 >= 5 and yd2 - yd1 >= 5:
                            boxes_disp.append((xd1, yd1, xd2, yd2))
                            cv2.rectangle(base_disp, (xd1, yd1), (xd2, yd2), (0, 0, 255), 2)
                            cur_disp = base_disp.copy()

                cv2.namedWindow("Select")
                cv2.setMouseCallback("Select", mouse_cb)
                while True:
                    cv2.imshow("Select", cur_disp)
                    k = cv2.waitKey(20) & 0xFF
                    if k == 13:  # Enter
                        break
                    elif k == 27:  # Esc → 取消這張
                        boxes_disp = []
                        break
                cv2.destroyWindow("Select")

                # 把顯示座標轉回原圖座標
                boxes_orig = []
                for (xd1, yd1, xd2, yd2) in boxes_disp:
                    rx1, ry1 = int(xd1/scale), int(yd1/scale)
                    rx2, ry2 = int(xd2/scale), int(yd2/scale)
                    if rx2 > rx1 and ry2 > ry1:
                        boxes_orig.append((rx1, ry1, rx2, ry2))
                if not boxes_orig:
                    continue

                to_paste = []
                for i, (x1,y1,x2,y2) in enumerate(boxes_orig,1):
                    crop = img_np_full[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    # 縮放長邊 ≤1080
                    ch, cw = crop.shape[:2]
                    s = 1080.0 / max(ch,cw) if max(ch,cw)>1080 else 1.0
                    crop_resized = cv2.resize(crop,(int(cw*s),int(ch*s)), interpolation=cv2.INTER_AREA)

                    text_detected = vlm_recognize_text_from_bgr(crop_resized)

                    if not text_detected:
                        put_log(f"[{idx_img}/{len(filepaths)}][區域 {i}] VLM 無結果，略過。\n")
                        continue

                    try:
                        zh = gguf_translate_text_to_chinese(text_detected) if use_context else gguf_translate_text_direct(text_detected)
                    except Exception:
                        zh = text_detected

                    put_log(f"[{idx_img}/{len(filepaths)}][區域 {i}] {text_detected} => {zh}\n","zh_color")

                    try:
                        vertical_mode = orientation_var.get().startswith("vertical")
                    except Exception:
                        vertical_mode = True

                    _auto_mask = build_text_mask_from_crop_auto(crop, dilate_px=3)
                    _crop_inpaint = inpaint_crop_with_mask(crop, _auto_mask, radius=3, method="telea")

                    while True:
                        prev_bgr = render_preview_on_inpainted_crop(_crop_inpaint, zh, vertical_mode)
                        cv2.imshow("Preview", prev_bgr)
                        k = cv2.waitKey(20) & 0xFF
                        if k in (ord('y'),ord('Y')):
                            to_paste.append((x1,y1,x2,y2, zh, vertical_mode))
                            cv2.destroyWindow("Preview")
                            break
                        elif k in (ord('n'),ord('N'),27):
                            cv2.destroyWindow("Preview")
                            break
                        elif k in (ord('v'),ord('V')):
                            vertical_mode = True
                        elif k in (ord('h'),ord('H')):
                            vertical_mode = False

                for (x1,y1,x2,y2,zh,vertical_mode) in to_paste:
                    paste_inpaint_then_text_using_box(img_pil_full,(x1,y1,x2,y2),zh,vertical_mode)

                base = os.path.splitext(os.path.basename(filepath))[0]
                out_path = os.path.join(SAVE_DIR,f"{base}_translated.png")
                if to_paste:
                    img_pil_full.save(out_path)
                    put_log(f"[{idx_img}/{len(filepaths)}] 【完成】已輸出：{out_path}\n","zh_color")
                else:
                    put_log(f"[{idx_img}/{len(filepaths)}] 未輸出圖片。\n")
            except Exception as e:
                put_log(f"[{idx_img}/{len(filepaths)}] 處理失敗: {e}\n")
        put_log("【批次完成】所有選取圖片皆已處理。\n","zh_color")

    threading.Thread(target=_worker,daemon=True).start()

def check_threads_end():
    # 若 record_thread 和 transcribe_thread 均為 None 或不存活，則認為已停止
    if ((record_thread is None) or (not record_thread.is_alive())) and \
       ((transcribe_thread is None) or (not transcribe_thread.is_alive())):
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 程式已停止。\n")
    else:
        root.after(100, check_threads_end)

def stop_transcription():
    transcribe_flag.set()
    # 不需推送 None 給 processing_queue，處理線程會根據 flag 結束
    check_threads_end()
    close_audio_stream()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = tk.Button(
    button_frame, text="開始轉錄", command=start_transcription,
    width=15, bg='green', fg='white'
)
start_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(
    button_frame, text="停止轉錄", command=stop_transcription,
    width=15, bg='red', fg='white', state=tk.DISABLED
)
stop_button.pack(side=tk.LEFT, padx=10)

upload_btn = tk.Button(
    button_frame, text="上傳圖片翻譯",
    command=lambda: upload_and_translate_image_vlm() if use_ollama_var.get() else upload_and_translate_image(),
    width=18
)
upload_btn.pack(side=tk.LEFT, padx=10)
# ========== 讀寫記憶 ==========

def load_short_term_memory():
    """
    從 JSON_MEMORY_FILE 載入短期記憶 (list of dicts)。
    如果檔案不存在或格式錯誤，返回預設記憶 (只有 system)。
    """
    if not os.path.exists(JSON_MEMORY_FILE):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一個轻小说翻译模型，可以流畅通顺地将日文、英文翻译成简体中文，"
                                "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。\n"
                    }
                ]
            }
        ]

    try:
        with open(JSON_MEMORY_FILE, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
        return memory_data
    except:
        # 若檔案損壞或解析失敗，也回傳預設
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一個轻小说翻译模型，可以流畅通顺地将日文、英文翻译成简体中文，"
                                "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。\n"
                    }
                ]
            }
        ]

def save_short_term_memory(memory_data):
    """
    將短期記憶 (list of dicts) 寫回 JSON_MEMORY_FILE。
    """
    with open(JSON_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=2)

def on_close():
    global record_thread, transcribe_thread
    transcribe_flag.set()
    for t in (record_thread, transcribe_thread):
        try:
            if t is not None and t.is_alive():
                t.join(timeout=1.5)
        except Exception:
            pass
    close_audio_stream()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# 開始 GUI 更新循環
root.after(100, update_gui)
root.mainloop()
