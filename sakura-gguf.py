import pyaudio
import numpy as np
import threading
import queue
import torch
import time
import webrtcvad
from datetime import datetime
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
import json
import os
import resampy
import soundfile as sf

from demucs.pretrained import get_model as demucs_get_model
from demucs.apply import apply_model as demucs_apply_model

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

from llama_cpp import Llama

# ========== 視窗固定 ==========
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

paddleocr_model = None  # 全局變量
persistent_window = None  # 全局變量，記錄持久OCR窗口對象
persistent_ocr_active = False  # 標記是否正在進行持續OCR
def load_paddleocr():
    global paddleocr_model
    if paddleocr_model is None:
        from paddleocr import PaddleOCR
        # 根據需要調整 lang 參數，如 "japan", "en", "ch" 等
        paddleocr_model = PaddleOCR(use_angle_cls=True, lang="japan")
        
record_thread = None
transcribe_thread = None

# -----------------------------------------------------
# 1. 載入 Qwen2.5 翻譯模型
# -----------------------------------------------------
qwen_gguf_path = "./sakura-7b-qwen2.5-v1.0-q6k.gguf"

print("使用 llama-cpp-python 加載 Qwen2.5 GGUF 模型中，請稍候...")
llm = Llama(
    model_path=qwen_gguf_path,
    n_ctx=4096,           # 上下文長度
    temperature=0.6,
    top_p=0.95,
    repeat_penalty=1.1,
    n_gpu_layers=30,      # 看模型大小去測試
    f16_kv=True,          
)
print("GGUF Qwen2.5 模型載入完畢。")

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
    使用短期記憶 (JSON) + llama-cpp (GGUF Qwen2.5)
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
    不使用記憶，單純呼叫 llama-cpp (GGUF Qwen2.5).
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

    # 仍然呼叫 llm() 做推理，stop 標記可以暫時維持不變
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
    if rounds > MAX_HISTORY_ROUNDS:
        conv_part = conv_part[2:]
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
    hf_whisper_path = "./turbo"
    device_idx = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_whisper_path,
        torch_dtype=torch_dtype,
        local_files_only=True
    )
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

# -----------------------------------------------------
# 3. 其餘參數 & GUI
# -----------------------------------------------------

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  #16kHz
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)   # 480
VAD_SENSITIVITY = 1
RMS_THRESHOLD = 400    # 音量閾值
MAX_SILENCE_CHUNKS = 5 # 若連續5個chunk無聲 => 結束錄音

RECORD_SECONDS_LIMIT = 10  # 最長錄幾秒避免拖太長

audio = pyaudio.PyAudio()
virtual_device_name = "CABLE Output"   #耳機輸出
device_index = None
for i in range(audio.get_device_count()):
    devinfo = audio.get_device_info_by_index(i)
    if virtual_device_name.lower() in devinfo['name'].lower():
        device_index = i
        break

if device_index is None:
    print(f"未找到音頻裝置: {virtual_device_name}")
    exit(1)

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
    exit(1)

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
root.geometry("600x900")

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

context_var = tk.BooleanVar(value=True)
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
        rms_val = compute_rms(chunk)
        if rms_val < RMS_THRESHOLD:
            continue

        # ========== 一旦音量>=閾值 => 進入'錄音'狀態 ==========
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 音量{rms_val:.0f}>=閾值 {RMS_THRESHOLD},開始錄...\n")

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
            rms_val = compute_rms(vocals_int16)
            if rms_val < 200:     # 確認分離後有聲音才傳給whisper
                gui_queue.put(f"無人聲跳過\n")
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
    # 將截圖傳給 PaddleOCR
    temp_file = "temp_ocr.png"
    screenshot.save(temp_file)
    # OCR處理
    from paddleocr import PaddleOCR
    # 根據用戶選擇語言，可動態調整 PaddleOCR 的 lang 參數，例如：若 language_var.get() == "english"，則 lang="en"
    if language_var.get() in ["japanese"]:
        ocr_lang = "japan"
    if language_var.get() in ["english"]:
        ocr_lang = "en"
    if language_var.get() in ["chinese"]:
        ocr_lang = "ch"
    ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang)
    result = ocr.ocr(temp_file, cls=True)
    ocr_text = ""
    for line in result:
        ocr_text += " ".join([w[1][0] for w in line]) + "\n"
    if not ocr_text.strip():
        gui_queue.put("OCR 為空，跳過翻譯\n")
        os.remove(temp_file)
        return
    # 顯示 OCR 原文
    gui_queue.put((f"[{datetime.now().strftime('%H:%M:%S')}] OCR 原文:\n{ocr_text}\n", "ja_color"))
    # 翻譯 OCR 結果（可選擇使用上下文記憶）
    # 翻譯 OCR 結果
    if context_var.get():
        translated = gguf_translate_text_to_chinese(ocr_text)
    else:
        translated = gguf_translate_text_direct(ocr_text)
    gui_queue.put((f"[{datetime.now().strftime('%H:%M:%S')}] 翻譯結果:\n{translated}\n", "zh_color"))
    os.remove(temp_file)

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
    if not persistent_ocr_active or not persistent_window:
        return  # 已停止或窗口不存在
    # 取得當前幾何
    x, y, w, h = persistent_window.get_region()
    import pyautogui
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    temp_file = "temp_ocr.png"
    screenshot.save(temp_file)

    # 動態設置語言
    if language_var.get() == "japanese":
        ocr_lang = "japan"
    elif language_var.get() == "english":
        ocr_lang = "en"
    elif language_var.get() == "chinese":
        ocr_lang = "ch"
    if paddleocr_model is None:
        load_paddleocr()
    result = paddleocr_model.ocr(temp_file, cls=True)
    ocr_text = ""
    for line in result:
        ocr_text += " ".join([w[1][0] for w in line]) + "\n"
    os.remove(temp_file)
    
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

    # 再次調度下次OCR
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
            if isinstance(msg, tuple) and len(msg) == 2:
                # 用於標色顯示
                text, tag = msg
                gui_text.insert(tk.END, text, tag)
            else:
                gui_text.insert(tk.END, msg)
            gui_text.see(tk.END)  # 確保每次插入後都滾動
        except queue.Empty:
            pass
    root.after(100, update_gui)  # 每100ms檢查一次

def start_transcription():
    global record_thread, transcribe_thread, voice_models_loaded
    transcribe_flag.clear()
    gui_text.delete(1.0, tk.END)
    current_mode = mode_var.get()
    if current_mode == MODE_VOICE:
        # 動態加載語音相關模型（Whisper、Demucs）只在首次選擇時加載
        if not voice_models_loaded:
            load_voice_models()  # 載入
        record_thread = threading.Thread(target=record_audio, daemon=True)
        transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)
        record_thread.start()
        transcribe_thread.start()
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 語音轉錄模式啟動。\n")
    else:
        # OCR 模式：延遲加載 PaddleOCR 模型
        load_paddleocr()
        gui_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] OCR 模式啟動\n請點擊 '選取區域' 按鈕進行單次辨識\n'持續辨識' 按鈕進持續辨識\n右鍵拖拽可改區域大小\n左鍵拖拽可移動\n")
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

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
    # 若 record_thread 為 None 或未啟動，就不做任何動作
    if record_thread is not None and record_thread.is_alive():
        transcribe_flag.set()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# 開始 GUI 更新循環
root.after(100, update_gui)
root.mainloop()

try:
    stream.stop_stream()
    stream.close()
    audio.terminate()
except Exception as e:
    print(f"關閉音頻流時出錯: {e}")
