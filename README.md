# AnimateDiff
AnimateDiff Motion Model 
最終目標，做出類似絲滑的短片

pixai-1849277974390037256(1).mp4
1. AnimateDiff 簡介
AnimateDiff 是在 Stable Diffusion 基礎上，加入了「時間序列」的考量，使生成多幀圖像時保有較高的一致性與連續性，減少幀與幀之間的亂跳或變形。它通常需要：
Stable Diffusion 的模型權重 (例如 v1.5、SDXL 等)
AnimateDiff Motion Model (專門處理動態的一組額外權重)
一段程式碼 來執行生成多幀流程
基本原理：在擴散過程中，除了空間維度 (xy) 之外，還加入「時間維度 (t)」的潛變量控制，讓整個生成同時考慮連貫性。

2. 事前準備
2.1 硬體需求
NVIDIA GPU (VRAM 8GB 以上較佳)：
Colab 通常會提供 Tesla T4 或以上的 GPU。
Windows 本地則需要一張 RTX 系列 (3060 / 3070 / 3080 …) 或更高。
系統環境
Colab：只要有 Google 帳號即可使用，但要注意免費版本有運算時間/規格限制。
Windows：需安裝 CUDA、Python 等，並保證 GPU 能夠正常驅動 Deep Learning 框架 (PyTorch 等)。
2.2 軟體套件
Python 3.8+
Colab 已內置 Python 3.
Windows 要自行安裝 (建議用 Python.org 的 64-bit 版本，或安裝 Anaconda 也行)。
PyTorch
Colab 上通常內置。
Windows 本地可按照 PyTorch 官網 說明，安裝對應 GPU/CUDA 版本的 PyTorch。
Git (Windows)：
建議安裝 Git 方便克隆專案 (若用 zip 也可，只是手動麻煩些)。
其他 Python 套件
例如 transformers, diffusers, safetensors, numpy, opencv-python, tqdm, einops, ftfy, accelerate, xformers 等，之後根據專案需求安裝。

3. 在 Colab 安裝與運行 AnimateDiff
3.1 建立一個全新的 Colab 筆記本
打開 Google Colab，選擇「新建筆記本」。
點選「修改」→「筆記本設定」，確保 硬體加速器 選擇 GPU。
3.2 安裝必要套件與下載 AnimateDiff 專案
在第一個程式碼儲存格，執行以下命令（範例）：
python
複製編輯
!nvidia-smi  # 檢查 GPU 狀態

# 1. 安裝Git（Colab通常內建，如果沒有就先安裝）
!apt-get update
!apt-get install git

# 2. clone AnimateDiff repo (以 guoyww/AnimateDiff 為例)
!git clone https://github.com/guoyww/AnimateDiff.git
%cd AnimateDiff

# 3. 安裝需求 (requirements.txt 或者 repo的README內指令)
!pip install -r requirements.txt

# 有些套件不在requirements裡，就手動安裝
!pip install xformers accelerate safetensors opencv-python

有些時候 AnimateDiff 的版本會更新，你可查看該 repo 的 README，若有最新安裝步驟，就以官方指令為準。
3.3 下載 / 放置模型權重
(1) Stable Diffusion 基礎模型
你可以使用 v1-5-pruned.ckpt 或 sd-v1-5.safetensors。
也可用 SDXL，但 AnimateDiff 目前對 SDXL 的支援還在演進中，可能需要特定分支。
下載方式：
從 Hugging Face Model Hub 或 CIVITAI 下載檔案。
放到當前專案指定的資料夾 (通常是 ./models/Stable-diffusion/) 或 ./checkpoints/ (依照 repo 要求)。
在 Colab 上可以用 wget 或 curl 命令直接下載到雲端硬碟，也可用 gdown if 是 Google Drive 連結。
範例（假設使用 v1.5 safetensors 檔）：
python
複製編輯
%cd /content/AnimateDiff
!mkdir -p models/Stable-diffusion
%cd models/Stable-diffusion

# 從 HuggingFace 下載 (用自己的權重連結)
!wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors

(2) AnimateDiff Motion Model
AnimateDiff 需要一個專門的 Motion Model 權重 (有些 repo 會在 README 提供下載連結)。
下載之後放到 repo 指定的 ./models/Motion_Model/ 或類似資料夾。
例如：motion_module_sd15.ckpt 或 motion_model_safetensors.
若在 repo 看到教學，如 wget https://huggingface.co/xxxx/yyy，跟著做即可。假設放在 ./models/Motion_Model/ 裏。

3.4 執行官方範例 (Demo Notebook or Script)
在 AnimateDiff repo 通常會有一個 demo.ipynb 或 scripts/animate.py 等示例。
你可以 cd 到那個目錄後，執行 python scripts/animate.py --prompt "xxxx" …
或直接打開「demo.ipynb」在 Colab 中執行，看官方範例怎麼跑。
範例指令 (純示意，不一定跟官方一模一樣)：
python
複製編輯
%cd /content/AnimateDiff
!python scripts/animate.py \
    --prompt "A little girl dancing, dynamic motion, masterpiece" \
    --motion_module "./models/Motion_Model/motion_module_sd15.ckpt" \
    --ckpt "./models/Stable-diffusion/v1-5-pruned.safetensors" \
    --num_frames 12 \
    --width 256 \
    --height 256 \
    --guidance_scale 7.5 \
    --seed 1234

參數說明：
prompt: 文字描述
motion_module: 動態模型路徑
ckpt: SD 主模型路徑
num_frames: 要生成幾幀
width / height: 影像大小
guidance_scale: 通常 7~12 可試
seed: 指定隨機種子以便複現
執行完後，它應該會在 outputs/ 或指定路徑下存 PNG 幀。可能會自動幫你合成 GIF 或 MP4，也可能需要額外指令；若需自己合成，可看下一節。

3.5 合成影片 (若範例沒自動合成)
常見做法是用 FFmpeg。若 AnimateDiff 腳本本身沒合成影片，可以手動執行：
bash
複製編輯
!apt-get install ffmpeg  # 如果Colab尚未安裝

# 假設 PNG 幀都在 outputs/animation/ 目錄下
!ffmpeg -framerate 8 -i outputs/animation/frame_%03d.png -c:v libx264 -pix_fmt yuv420p output.mp4

-framerate 8：每秒 8 幀；你可以改成 12、24 等。
frame_%03d.png：對應影格檔案命名。
-c:v libx264 -pix_fmt yuv420p：常見的 H.264 編碼、保證通用播放。
合成完後，可以在 Colab 左側檔案樹下載 output.mp4 或直接用 from IPython.display import Video 在 notebook 播放。

4. 在 Windows 本地安裝與運行 AnimateDiff
若你想在 Windows 上本地跑，流程類似，但需要手動安裝與配置：
4.1 安裝 NVIDIA 驅動、CUDA、Python、Git
更新顯示卡驅動
從 NVIDIA 官網 下載最新版驅動 (或透過 GeForce Experience)。
安裝 CUDA Toolkit (可選)
通常 PyTorch 只需要對應的 CUDA runtime，如 CUDA 11.7；若你想開發或編譯，也可安裝完整 CUDA Toolkit。
安裝 Python 3.8 或 3.9
可以用 Python.org 的 installer，或 Anaconda (Miniconda)。
安裝 Git
從 Git 官網 下載，安裝以便 clone repo。
4.2 Clone AnimateDiff 專案 & 安裝套件
打開 CMD 或 PowerShell (或 Anaconda Prompt)，執行：
bash
複製編輯
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff
pip install -r requirements.txt
pip install xformers accelerate safetensors opencv-python

如果報錯，就根據提示安裝缺的套件或更新 pip 版本，如 pip install --upgrade pip。
4.3 下載模型 (SD + Motion Model)
在檔案總管中，建立資料夾 AnimateDiff/models/Stable-diffusion。
將你的 Stable Diffusion 權重檔 (e.g. v1-5-pruned.safetensors) 放進去。
在 AnimateDiff/models/Motion_Model/ 放入 AnimateDiff motion model (e.g. motion_model_sd15.safetensors 之類)。
若 AnimateDiff 要特定命名或路徑，依照 README 做。
4.4 執行測試腳本
進入 AnimateDiff 資料夾後，執行：
bash
複製編輯
python scripts/animate.py ^
  --prompt "A dancing cat with disco lights, cartoon style" ^
  --motion_module "./models/Motion_Model/motion_model_sd15.safetensors" ^
  --ckpt "./models/Stable-diffusion/v1-5-pruned.safetensors" ^
  --num_frames 12 ^
  --width 256 ^
  --height 256 ^
  --guidance_scale 8

(Windows CMD 用 ^ 換行, Powershell 或 Git Bash 可用 \ 換行)
這會在命令列顯示推論過程。結束後，在 outputs/ (或指定位置) 找到輸出的多張 PNG / MP4 / GIF。若沒有合成影片，就手動用 ffmpeg (同上做法)。

5. 進階調校與技巧
num_frames (幀數)
越多幀越耗時；初期可先測試 8~12 幀看看。
解析度 (width, height)
建議 256x256、512x512 起步。解析度越高，VRAM 和運算時間越多。
guidance_scale
7~12 之間調整，越高越貼近 Prompt，但可能抖動或失真。
采樣 steps
20~30 是常見範圍。
seed
固定 seed 可復現同樣結果；若想要更多隨機性，就不指定 seed 或設 -1。
其他動作參數
AnimateDiff 有些設定用來平衡「幀間一致」vs.「畫面多樣性」。可參考官方 README，調整 motion alpha / fps / scheduler (如 Euler_A, DDIM, DPM++ 2M, Karras 等)。

6. Colab vs. Windows：哪個更容易？
Colab 優勢
不用本機裝 CUDA，只要瀏覽器 + Google 帳號，即可立即測試。
GPU 資源免费（有一定使用限制）。
輕鬆分享 notebook。
Colab 劣勢
連線時間有限 (12 小時到期或不定時斷線)。
免費用戶可能拿到較慢或較小的 GPU，生成速度有限。
大檔案（模型）每次都要重新下載或放 Google Drive 連結。
Windows 優勢
隨時可用，不中斷；GPU 如果夠強，就能穩定長期使用。
可以整合本地任何程式 (如 AUTOMATIC1111 WebUI, 其他 LoRA, ControlNet)。
Windows 劣勢
需要安裝 CUDA、PyTorch、Git…較繁雜。
如果顯卡太弱（<8GB VRAM），生成就會卡或OOM（Out of Memory）。
需要自己管理好Python環境、套件衝突問題。
結論
想快速試用、免安裝 → Colab 比較容易上手。
想長期大量生成、有較好 GPU → 在 Windows 本地環境最終能更靈活。

7. 常見問題 (FAQ)
報 CUDA out of memory？
降低解析度 (width/height)
降低幀數 (num_frames)
減少 batch size 或將 --precision full 改成 --precision autocast 等等。
生成結果會大幅閃爍或跳動？
嘗試不同的 scheduler (e.g. DDIM, Euler_A, DPM++ 2M Karras)
降低 guidance_scale
在 prompt 中適當加入 consistent character, stable motion 等提示詞
生成速度太慢？
使用較好的 GPU (e.g. T4, V100, A100)
調降 sampling steps (20 or even 15)
減少幀數
找不到權重/路徑錯誤？
檢查指令參數，確保 --ckpt 路徑正確；或將模型檔名與脚本要求一致。
SDXL 能用 AnimateDiff 嗎？
目前多數 AnimateDiff 分支主要針對 SD 1.5，SDXL 的支援在逐步開發中。你需要找到專門支援 SDXL 的 fork，或自己動手改。

8. 簡要流程總結
準備環境 (Colab 或 Windows) → 安裝 Python、PyTorch、必需套件。
Clone AnimateDiff Repo → 安裝 requirements。
下載模型 (Stable Diffusion 1.5 + AnimateDiff Motion Model) → 放到指定資料夾。
執行示例脚本 / notebook → 填入 prompt、幀數、解析度等參數 → 生成 PNG 幀。
(可選) FFmpeg 合成影片 → 得到 MP4 / GIF。
調參優化 → 改變 sampling steps、guidance_scale、scheduler、prompt 詞彙，觀察結果。
這樣你就能在 Colab 或 Windows 下跑起「Stable Diffusion + AnimateDiff」，實現多幀連續、比較平順的角色或場景動畫生成。祝你玩得愉快、實驗順利！

