# Task 6 · MMCultureQA English textual QA

Pipeline riêng trong `D:\semeval\task 6`, không thay đổi Task 8. SFT và GRPO đều dùng **ms-swift 3.11.3 + PEFT**. Mã project chỉ chuyển cấu hình/dataset, đăng ký metric/reward và lưu adapter; không tự triển khai vòng tối ưu.

## Bắt đầu trên Colab

Bốn notebook dùng cùng một cấu hình lưu trữ. Bạn chỉ nhập ID (hoặc URL) của folder `MMCQA` vào `ROOT_FOLDER_ID`, và cung cấp file rclone đã xác thực qua `RCLONE_CONFIG_PATH`. Notebook tự chọn remote nếu config chỉ có một Google Drive remote; nếu nhiều remote thì chọn `REMOTE`. Không nhập đường dẫn MyDrive hay thư mục đích.

```python
ROOT_FOLDER_ID = 'ID_HOAC_URL_FOLDER_MMCQA'
RCLONE_CONFIG_PATH = '/content/rclone.conf'
REMOTE = ''  # tự chọn nếu có đúng một Google Drive remote
```

Cấu trúc bên dưới root được tự thiết lập, không phụ thuộc vị trí/tên đường dẫn mount của folder:

```text
MMCQA/                           ← ROOT_FOLDER_ID
├── task6-data/raw/               nhãn và 3 ZIP ảnh
└── task6-runs/<RUN_NAME>/
    ├── sft/                     hoặc sft-smoke
    │   ├── final/              adapter được chọn + tokenizer/processor
    │   ├── run.json            cấu hình, prompt, phiên bản, hash manifest
    │   ├── learning_curves.png và log/metadata nhỏ
    │   └── validation/         điểm, tóm tắt và đáp án validation
    ├── grpo/                    hoặc grpo-smoke
    │   └── ...                 cùng cấu trúc SFT
    └── predictions/             JSONL và scores
```

1. `01_hf_to_drive_rclone.ipynb`: tải từ Hugging Face trên Colab, upload nhãn/ZIP vào `task6-data/raw` bên dưới root.
2. `02_sft_colab.ipynb`: train SFT, upload adapter `final` và báo cáo cần thiết, sau đó **inference toàn bộ validation và chấm scorer chính thức**, lưu `validation/scores.json` và predictions lên Drive.
3. `03_grpo_colab.ipynb`: chỉ train GRPO tiếp từ SFT, base NF4 4-bit và Transformers generation, không tạo model vLLM thứ hai. Mặc định đọc `SFT_RUN_NAME='experiment01'`, ghi run mới `RUN_NAME='experiment01_qlora'`; chọn `INPUT_SFT_STAGE`, `SFT_CHECKPOINT` khi cần. Cuối cùng **inference toàn bộ validation và chấm scorer chính thức**, lưu kết quả riêng bên trong run GRPO.
4. `04_inference_colab.ipynb`: chỉ inference test (`devtest` mặc định), tự upload predictions. `INPUT_RUN_NAME` chọn SFT, `GRPO_INPUT_RUN_NAME` chọn run QLoRA mới. `ADAPTER_4BIT` chọn lượng tử hóa từng adapter (mặc định SFT False, GRPO True). Nếu không có nhãn công khai thì chỉ xuất file nộp, không tính điểm local. Không dùng notebook này để chấm validation.

Mỗi notebook clone repo `iamtro-2006/semeval` và cài package trong `task 6/`. Không ZIP/upload code. Dùng cùng `ROOT_FOLDER_ID` trong các phiên để đồng bộ. File rclone gốc không bị sửa; dùng bản sao tạm để refresh token. Nếu config nằm trên Drive thì chạy cell mount tùy chọn chỉ để đọc config.

Mặc định Colab làm việc theo cấu trúc sau, kể cả khi mở notebook từ GitHub và kernel bắt đầu tại `/content`:

```text
/content/
├── rclone.conf                  config đã xác thực, upload riêng
└── semeval/                     CHECKOUT: repo clone
    └── task 6/                  PROJECT_ROOT
        ├── src/mmcultureqa/     module Python
        ├── configs/             cấu hình SFT/GRPO/inference
        ├── data/raw/            dữ liệu tải từ Drive/Hugging Face
        ├── data/prepared/       manifest có đường dẫn ảnh tương đối với project
        └── outputs/            adapter đầu vào, run và predictions local
```

Setup xác định `PROJECT_ROOT` từ `CHECKOUT`, thêm `src` vào `sys.path` của kernel và `PYTHONPATH` cho subprocess/worker, xóa cache module project cũ rồi kiểm tra import ngay. Mọi lệnh Python dùng `sys.executable`; launcher cố định thư mục chạy ở `PROJECT_ROOT`. Đường dẫn config, run và adapter trong notebook là tuyệt đối, không phụ thuộc cell sau có đổi thư mục làm việc. `CHECKOUT` và `RCLONE_CONFIG_PATH` phải là đường dẫn tuyệt đối; không đặt config vào repo.

Inference dùng launcher riêng, kiểm tra adapter đầu vào và upload predictions sau mỗi stage thành công. Thư mục predictions không cần chứa `final/` của training. Sau khi cập nhật notebook, chạy lại từ cell setup; nếu vừa thay dependency nhị phân thì restart runtime rồi chạy lại từ đầu.

Mặc định `RUN_FULL=True` chạy đầy đủ; đổi thành `RUN_FULL=False` để thử một optimizer step. Khi chạy GRPO đầy đủ, chọn adapter SFT đầy đủ. `final` chứa adapter + processor, không chứa bản sao toàn bộ Qwen. Resume cùng stage dùng `RESUME_CHECKPOINT='checkpoint-N'` chỉ khi đã tự sao lưu checkpoint đầy đủ lên Drive, giữ nguyên cấu hình; chuyển SFT sang GRPO dùng adapter và optimizer GRPO mới.

Đánh giá cuối trong hai notebook train mặc định luôn dùng **đủ 1.000 mẫu validation**, độc lập với số mẫu/step smoke training. `VAL_LIMIT=None` là full; đổi thành `4` chỉ để thử pipeline đánh giá, kết quả subset được lưu tên riêng, không gọi là full-val. Scorer báo BERTScore-F1/P/R, BLEU, ROUGE-L và Coverage; đây không phải tỷ lệ exact-match accuracy. Đánh giá cuối không thay thế monitoring định kỳ: SFT mặc định chọn checkpoint bằng val loss; bật generation thì chọn bằng val BERTScore-F1. GRPO full run mặc định chọn checkpoint có val reward cao nhất theo cơ chế ms-swift.

Chạy lại đánh giá adapter cuối từ CLI, không train lại:

```bash
python -m mmcultureqa.evaluate --run-dir outputs/sft
python -m mmcultureqa.evaluate --run-dir outputs/grpo
```

Evaluator dùng prompt/model settings đã lưu trong `run.json`, kiểm tra manifest không đổi và train/val không trùng ID/ảnh, sinh đáp án rồi chấm scorer trong hai subprocess để giải phóng VRAM giữa VLM và mmBERT. Chỉ đọc gold để chấm; gold không đưa vào prompt.

Training và ảnh dùng SSD local; checkpoint trung gian chỉ lưu local. Rclone dùng danh sách cho phép trong `src/mmcultureqa/storage.py`, chỉ upload các mục sau:

| Mục giữ trên Drive | Mục đích |
|---|---|
| `final/` trừ `training_args.bin` | Adapter được chọn và tokenizer/processor để dùng lại |
| `run.json`, `swift_arguments.json` | Cấu hình thực tế, prompt, phiên bản và thông tin dataset |
| `logging.jsonl`, `trainer_state.json`, `learning_curves.png` nếu có | Xem tiến trình học và lựa chọn checkpoint |
| `validation/` hoặc `validation_subset_N/`: `scores.json`, `evaluation.json`, `predictions.jsonl` | So sánh kết quả và xem các câu trả lời sai |
| `predictions/`: JSONL SFT/GRPO cho devtest/test, cấu hình inference và scores nếu có nhãn | File nộp bài và thông tin truy vết |
| `task6-data/raw/`: 3 JSONL English, 3 ZIP ảnh, `REVISION` | Một bản dữ liệu dùng chung cho mọi run |

Không upload checkpoint trung gian, optimizer/scheduler state, dataset đã export cho ms-swift, ảnh giải nén, cache, TensorBoard, Parquet trùng dữ liệu, gold validation/test sao chép, signature và config tạm của evaluator. File mới phát sinh ngoài danh sách cũng ở lại local. Những báo cáo nhỏ được giữ để không phải chạy lại model chỉ để xem kết quả; các lần `rclone copy` sau bỏ qua file không đổi.

Nếu Colab mất runtime trước khi upload cuối hoàn tất, checkpoint local không có trên Drive để phục hồi. Full run mặc định chọn SFT best theo val loss và GRPO best theo val reward; chế độ smoke không chọn best theo validation. Remote dùng `copy`, không xóa file cũ đã có trên Drive; `save_total_limit` chỉ giới hạn checkpoint local. CLI ngoài notebook không tự upload.

Sửa notebook generator rồi sinh lại:

```bash
python tools/build_notebooks.py
```

Kiểm tra hồi quy đường dẫn/import và launcher trên CPU (từ thư mục repo):

```bash
python -m unittest discover -s "task 6/tests" -v
```

Các kiểm tra mô phỏng layout `content/semeval/task 6` trong thư mục tạm, chạy import thật ở kernel/Python con sau khi đổi working directory, và giả lập train/inference/rclone để kiểm tra đường dẫn cùng upload. Chúng không thay thế lượt chạy GPU Colab với các dependency runtime.

## Dữ liệu đã kiểm tra thực tế

Dataset: https://huggingface.co/datasets/QCRI/MMCQA-SemEval27

Pinned revision: `e850ede73e2f1ef3dde12e0210de67d107247ef0`. Subset chính xác là `qa_mena_en`. Chỉ MENA có English QA ở snapshot này. Số task ngoài cuộc thi có thể đổi; tên thư mục local theo yêu cầu là Task 6, còn organizer phân biệt textual QA và spoken QA.

| Split nguồn | Mẫu | Nhãn | Ảnh |
|---|---:|---|---|
| train | 10.000 | Có answer | images_mena_train.zip |
| dev | 1.000 | Có answer | images_mena_dev.zip |
| devtest | 1.000 | Không có answer | images_mena_devtest.zip |

Các JSONL và Parquet là hai biểu diễn cùng dữ liệu; ảnh là đường dẫn `images/<id>.jpg`, không nhúng trong Parquet chính. Không lấy demo 24 mẫu thay cho full dataset; không tải SQA/audio hay các ngôn ngữ QA khác. English chỉ là ngôn ngữ câu hỏi/đáp án, chữ trong ảnh vẫn có thể là Arab hoặc đa ngôn ngữ.

`prepare` kiểm tra duplicate ID, overlap ảnh/ID giữa split, file ảnh trong ZIP, CRC và giải mã cấu trúc ảnh. Tách train theo nhóm ảnh với seed 42 thành 9.000 train và 1.000 validation; dev giữ riêng cho đánh giá, devtest chỉ inference. Manifest dùng đường dẫn tương đối để chuyển sang Colab. Xem `docs/DATA_INSPECTION.json` và `docs/DATASET.md`.

## Cấu trúc

```text
semeval/
├── task 8/
└── task 6/
    ├── pyproject.toml          package và CLI entry points
    ├── requirements.txt       dependency runtime được pin
    ├── configs/               sft.yaml, grpo.yaml, inference.yaml
    ├── prompts/               prompt dùng chung, chỉnh được
    ├── src/mmcultureqa/
    │   ├── common.py          config, paths, prompt và provenance
    │   ├── data/prepare.py    download, kiểm tra ảnh và chia split
    │   ├── models/loading.py  base model, processor và PEFT
    │   ├── training/          sft.py, grpo.py, swift_backend.py, swift_metrics.py
    │   └── inference/         predict.py
    ├── notebooks/             rclone transfer và Colab experiments
    ├── tools/                 notebook generator
    ├── docs/                  dataset, nguồn tham khảo và validation
    ├── vendor/organizer/      scorer chính thức + provenance
    ├── data/raw/              nhãn, ZIP và ảnh (không commit)
    ├── data/prepared/         manifests (không commit)
    └── outputs/               checkpoint/prediction (không commit)
```

## Chạy bằng CLI

Chạy từ thư mục `task 6` với Python 3.11 hoặc 3.12. Cần GPU CUDA cho train/inference; download/prepare chạy CPU. Dependency được pin trong `requirements.txt`. Không cài requirements CPU của organizer chồng vào GPU environment.

```bash
pip install -r requirements.txt
# Chỉ thêm nếu dùng BF16 standalone inference/vLLM:
pip install -r requirements-vllm.txt
pip install --no-deps -e .
python -m mmcultureqa.data.prepare download
python -m mmcultureqa.data.prepare prepare
python -m mmcultureqa.training.sft --config configs/sft.yaml
python -m mmcultureqa.training.grpo --config configs/grpo.yaml
python -m mmcultureqa.inference.predict --config configs/inference.yaml
python -m mmcultureqa.inference.predict --config configs/inference.yaml --set model.adapter=outputs/grpo/final --set model.load_in_4bit=true --set inference.backend=transformers --set inference.output=outputs/predictions/grpo_dev.jsonl
python -m mmcultureqa.inference.predict --config configs/inference.yaml --set model.adapter=outputs/grpo/final --set model.load_in_4bit=true --set inference.backend=transformers --set inference.split=devtest --set inference.output=outputs/predictions/grpo_devtest.jsonl
```

Ví dụ chỉnh tham số, prompt và resume:

```bash
python -m mmcultureqa.training.sft --config configs/sft.yaml --set model.load_in_4bit=true --set training.learning_rate=0.0001 --set prompt_file=prompts/qa.txt
python -m mmcultureqa.training.grpo --config configs/grpo.yaml --set model.load_in_4bit=true --set reward.device=cuda
python -m mmcultureqa.training.sft --config configs/sft.yaml --set resume_from_checkpoint=outputs/sft/checkpoint-100
```

SFT giữ mặc định **BF16 LoRA** (`USE_4BIT=False`). GRPO mặc định **QLoRA: base NF4 4-bit, double quantization, BF16 compute** (`USE_4BIT=True`); chỉ cập nhật adapter. GPU vẫn phải hỗ trợ BF16. mmBERT reward/monitor chạy CUDA FP16, batch 4. Không bảo đảm mọi Colab GPU đều đủ VRAM cho mọi cấu hình.

GRPO một GPU: train microbatch **4**, gradient accumulation **4**, `steps_per_generation=4`, rollout **16 completions = 4 câu hỏi × 4 đáp án**, eval batch **8**. Dữ liệu 9.000 câu/một epoch tương ứng khoảng **2.250 optimizer step** thay vì 9.000; mỗi step xử lý nhiều dữ liệu hơn nên số step giảm không đồng nghĩa thời gian giảm đúng 4 lần. Giữ nguyên learning rate 5e-6, số đáp án 4, reward, prompt và giới hạn generation 256 token. Batching được kiểm tra trước khi cấp phát model để tránh lỗi eval batch không chia hết cho số generation.

Nếu OOM, thử `GRPO_OVERRIDES = {'training.per_device_train_batch_size': 2, 'training.per_device_eval_batch_size': 4}`. Với `steps_per_generation=4`, rollout còn 8 completions; muốn giảm tiếp dùng train batch 1. Giữ eval batch chia hết cho 4 trên một GPU. Dùng run mới khi đổi recipe; không resume optimizer từ checkpoint BF16/vLLM cũ vào QLoRA mới. Adapter SFT BF16 vẫn được nạp qua cơ chế PEFT để train tiếp.

## Monitoring theo ms-swift

SFT mặc định `PREDICT_WITH_GENERATE=False`; GRPO bật validation rollout riêng. CLI dùng `training.predict_with_generate`.

Full run mặc định: SFT 3 epoch, GRPO 1 epoch; cả hai đánh giá và lưu checkpoint mỗi 170 optimizer step. SFT chọn val loss thấp nhất, GRPO chọn val reward cao nhất rồi xuất adapter đó vào `final/`. Tăng epoch vẫn giữ cơ chế chọn tốt nhất. `RUN_FULL=False` chỉ dành cho smoke một bước, không chọn checkpoint theo validation.

Cả hai notebook vẽ đường cong theo optimizer step trước bước đánh giá cuối: SFT train/val loss, GRPO train/val reward. Biểu đồ lưu thành `learning_curves.png` trong thư mục run và upload lên Drive. Nếu chưa tới kỳ evaluation thì chưa có điểm validation để vẽ.

| Stage/chế độ | Monitor định kỳ | Chọn checkpoint |
|---|---|---|
| SFT, False | train loss/token_acc; val loss/token_acc mỗi eval_steps | val loss thấp nhất |
| SFT, True | train loss/token_acc; sinh đáp án val và eval_bertscore_precision/recall/f1 | val BERTScore-F1 cao nhất |
| GRPO, False | loss/reward và thống kê rollout train của ms-swift | checkpoint cuối |
| GRPO, True (mặc định GRPO) | thêm rollout val; eval_rewards/MMCQABERTScore/mean khi reward.type=mmbert | val reward cao nhất |

SFT generation evaluation thay thế lượt teacher-forcing validation của framework: không hứa có val loss đồng thời trong cùng lượt. Token accuracy không phải điểm QA tự sinh. GRPO luôn sinh rollout khi train dù flag False; không có cách hiểu token accuracy/val loss giống SFT. Điểm reward val của GRPO là trên các đáp án sampled theo cấu hình rollout, không phải phép chấm greedy cuối. Với reward exact, tên metric tương ứng là MMCQAExact, không gọi đó là BERTScore.

```bash
# Mặc định không generation
python -m mmcultureqa.training.sft --config configs/sft.yaml
# Bật native generation evaluation + plugin mmBERT
python -m mmcultureqa.training.sft --config configs/sft.yaml --set training.predict_with_generate=true
python -m mmcultureqa.training.grpo --config configs/grpo.yaml --set training.predict_with_generate=true
```

`monitor.limit=null` dùng toàn bộ val, độc lập `data.limit` của train. Nếu đặt giới hạn nhỏ hơn số mẫu val (ví dụ 256), SFT và GRPO lấy mẫu ngẫu nhiên không lặp bằng seed **42**, cố định cùng subset/thứ tự qua các lần eval khi manifest không đổi; không còn lấy các mẫu đầu tiên. Bộ sinh số ngẫu nhiên riêng không làm đổi trạng thái random của training. Giới hạn lớn hơn hoặc bằng số mẫu giữ đủ val theo thứ tự gốc. `monitor.max_new_tokens` và `monitor.device` chỉnh generation metric SFT; reward GRPO chỉnh trong `reward`. Trong smoke notebook vẫn tắt evaluation SFT định kỳ (một step), nhưng đánh giá cuối giữ nguyên. Bước đánh giá cuối của cả hai stage luôn sinh đáp án và chấm scorer chính thức, không phụ thuộc flag monitor.

## QLoRA GRPO và backend generation

- GRPO: `model.load_in_4bit=true`, `training.use_vllm=false`, `training.vllm_mode=null`, `sleep_level=0`, model/optimizer offload False. Native ms-swift `PtEngine.from_model_template` dùng chính model training để sinh, không load Qwen thứ hai. ms-swift yêu cầu mode phải là None khi tắt vLLM; launcher chuẩn hóa giá trị này sau overrides. Các tham số dung lượng/cache vLLM không được dùng trên nhánh Transformers.
- Vẫn có một encoder mmBERT FP16 riêng để tính reward. mmBERT được preload và cache; việc tắt vLLM không bỏ scorer. QLoRA giảm bộ nhớ trọng số base, không giảm mọi activation/KV cache xuống 4-bit.
- SFT giữ BF16 LoRA và native Transformers generation monitor. Chỉ generation monitor mới cần load scorer.
- Final validation GRPO dùng Transformers 4-bit; CLI evaluator cũng tự chọn Transformers khi run đã lưu `load_in_4bit=true`. Model được giải phóng trước khi subprocess scorer chính thức chạy.
- Standalone inference cho phép SFT BF16/vLLM và GRPO 4-bit/Transformers trong hai subprocess riêng. Model được giải phóng giữa các stage.
- `requirements.txt` không yêu cầu vLLM. Notebook GRPO dùng `INSTALL_VLLM=False`, kiểm tra import PtEngine/bitsandbytes mà không bắt buộc vLLM. SFT/inference dùng `requirements-vllm.txt` khi cần backend BF16/vLLM (`vllm==0.11.0`, torch 2.8.0). Package vLLM đã cài từ phiên cũ có thể còn đó nhưng GRPO không khởi tạo engine của nó.
- Đây là cấu hình chưa benchmark trên GPU. Không bảo đảm dưới 8 tiếng hoặc batch 4 luôn vừa VRAM. Đo thời gian sau warmup và peak VRAM; lượng tử hóa có thể thêm chi phí giải lượng tử dù tiết kiệm được offload và một bản model.

Các knobs trong YAML vẫn chỉnh được. Khi chọn 4-bit ở notebook, standalone inference/final validation dùng backend Transformers; recipe vLLM hiện tại chỉ hỗ trợ BF16.

## Điều kiện huấn luyện

- Dataset export sang schema native ms-swift: messages + images; GRPO có thêm solution chỉ cho reward. SFT dùng template qwen3_vl và loss_scale=default của ms-swift để mask user/system/image/padding. Không dùng custom SFT collator cũ.
- LoRA dùng target_regex ngôn ngữ, freeze_vit/aligner=True; khi tiếp tục adapter thì dùng cơ chế adapters của ms-swift, không khởi tạo adapter ngẫu nhiên thay thế. Tham số thực sự học ghi vào run.json.
- max_length=None, packing=False, strict=True. Không chủ động cắt ảnh/đáp án hoặc thay mẫu lỗi. MAX_PIXELS/MIN_PIXELS được truyền cho qwen-vl-utils theo cơ chế ms-swift.
- beta mặc định 0. Nếu bật beta>0 cho GRPO, cấu hình ref_adapters cùng adapter SFT theo tài liệu ms-swift để KL tham chiếu SFT, không phải base trước SFT.
- Plugin mmBERT pin model/revision/layer, IDF off, no rescaling, fast tokenizer và giới hạn 512. Empty answer nhận 0. Scorer cache theo cấu hình để tránh tải lại mỗi batch. Giữ nguyên scorer cuối chính thức trong vendor.
- native arguments được lưu trong swift_arguments.json. Khóa `swift` trong YAML cho phép override trực tiếp các tham số TrainArguments/RLHFArguments; framework sẽ báo lỗi nếu tên tham số không hợp lệ. Cấu hình native override có quyền ưu tiên, vì vậy chỉ đổi khi hiểu tác động.
- Adapter/checkpoint được ms-swift lưu; hook project chỉ chuẩn hóa base-model identifier trong metadata và đồng bộ Drive sau khi train thành công qua notebook launcher. Adapter `final` không chứa base weights. Checkpoint có optimizer/scheduler dùng để resume cùng stage.
- GRPO rollout và final validation dùng Transformers với QLoRA, cùng pinned base revision và adapter. Standalone BF16 inference vẫn có thể dùng VllmEngine; backend chọn qua inference.backend. SFT generation monitor dùng Transformers theo cơ chế native ms-swift.

## Dùng adapter SFT đã công bố

```bash
python -m mmcultureqa.training.grpo --config configs/grpo.yaml --set model.adapter=anhbilong/qwen3-vl-4b-mmcultureqa-split512 --set model.adapter_revision=e62fdf28389cdc11f45f295f63e3e4c354e8eabe
```

Adapter này có split train/validation riêng. Không coi validation do pipeline này mới sinh là unseen đối với adapter đó. Để đánh giá nội bộ công bằng, nhập đúng split manifest của adapter gốc; dev nguồn vẫn là tập đối chứng riêng theo audit của repo tham khảo.

## Chấm điểm

```bash
python vendor/organizer/scorer/score.py --gold data/raw/qa/mena/dev_en.jsonl --pred outputs/predictions/sft_dev.jsonl --out outputs/sft_scores.json
```

Scorer xếp hạng bằng mmBERT BERTScore-F1; các file trong vendor giữ nguyên. Để khớp dependency leaderboard hoàn toàn, dùng environment CPU riêng với `vendor/organizer/requirements.txt`. Không chấm devtest vì thiếu nhãn. Dự đoán phải đầy đủ split; không so smoke subset với full gold.

## Trạng thái xác minh

Môi trường training pin `numpy==2.2.6` để hỗ trợ Python 3.13 và tránh xung đột với JAX trên Colab. Không tắt backend TensorFlow/JAX. Sau cài đặt, notebook kiểm tra import Transformers, PEFT, ms-swift/PtEngine, bitsandbytes, qwen-vl-utils và decord trước khi tải dữ liệu/train; chỉ kiểm tra vLLM khi `INSTALL_VLLM=True`. Lỗi sẽ in đầy đủ traceback. Sau khi đổi dependency nhị phân trên runtime cũ, restart session rồi chạy lại setup.

Giữ `decord==0.6.0` vì registry Qwen3-VL của ms-swift 3.11.3 khai báo dependency này; cảnh báo platform của wheel cần đối chiếu với kết quả import thực tế, không mặc định quy lỗi cho Python 3.13. Môi trường Colab có thể còn xung đột các package ngoài pipeline trong `pip check`; chưa có xác minh toàn bộ môi trường Colab/Python 3.13. Bộ requirements scorer organizer trong `vendor/` giữ nguyên và dùng môi trường riêng khi cần tái lập đúng dependency leaderboard.

Đã tải và kiểm tra đủ 12.000 ảnh local, đủ ba JSONL và ba Parquet. Đã kiểm tra cú pháp code/notebook, manifest và cấu hình trên CPU. Máy phát triển không có GPU: chưa chạy forward/backward Qwen3-VL, chưa xác nhận loss/gradient trên GPU, chưa chạy thực nghiệm Colab và chưa có kết quả chứng minh GRPO cải thiện. Notebook cung cấp lượt thử GPU để xác nhận trước full run.

Tham khảo API theo phiên bản được pin: `docs/SOURCES.md`.


### Precision của mmBERT

Mặc định `monitor.dtype=float16` và `reward.dtype=float16` trên CUDA. Scorer được cast FP16 một lần và cache; đổi dtype tạo cache riêng. Qwen SFT dùng BF16; Qwen GRPO lưu base NF4 với compute BF16. FP16 giảm bộ nhớ trọng số mmBERT và có thể tăng tốc; chưa benchmark tốc độ hoặc độ lệch điểm trên GPU.

Chấm cuối cũng dùng FP16 qua wrapper `python -m mmcultureqa.score`, giữ nguyên file organizer trong vendor. JSON điểm ghi `execution_precision`. Điểm accelerated FP16 có thể lệch nhẹ với FP32; muốn đối chiếu dùng `--dtype float32` hoặc chạy trực tiếp scorer organizer.

```bash
python -m mmcultureqa.score --dtype float16 --gold data/raw/qa/mena/dev_en.jsonl --pred outputs/predictions/sft_dev.jsonl --out outputs/scores_fp16.json
```
