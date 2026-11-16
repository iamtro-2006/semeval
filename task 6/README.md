# Task 6 · MMCultureQA · experiment03

Pipeline hiện tại ưu tiên SFT trên Qwen3-VL-4B-Instruct. Train bằng QLoRA 4-bit, chọn checkpoint bằng validation loss, sau đó xuất **một model BF16 hoàn chỉnh** để đánh giá và inference. Notebook GRPO tạm giữ nguyên recipe và tên run cũ; không chạy tự động trong experiment03.

## Cấu hình SFT

| Tham số | Mặc định |
|---|---|
| Run | `experiment03` |
| Base | `Qwen/Qwen3-VL-4B-Instruct`, revision pin trong YAML |
| Training weights / compute | NF4 4-bit + double quantization / BF16 |
| Epoch | 2 |
| Learning rate | Peak `1e-4`, warmup 3%, cosine decay |
| Batch training / accumulation | 8 / 1, effective batch 8 trên một GPU |
| Validation định kỳ | Teacher forcing: loss và token accuracy, không generation/mmBERT |
| Eval/save | Mỗi 225 optimizer step |
| Tập monitor | 256 mẫu val ngẫu nhiên không lặp, seed 42, cố định |
| Chọn best | Val loss thấp nhất |
| LoRA mặc định trong YAML | r=16, alpha=32; chỉnh qua SFT_OVERRIDES nếu muốn |
| Projector LoRA | Tùy chọn, mặc định tắt |
| Artifact trên Drive | `best_model/`: model BF16 đã merge + processor/tokenizer |

9.000 câu train × 2 epoch / batch 8 = 2.250 optimizer step, khoảng 10 kỳ validation. Đây không phải dự đoán thời gian GPU. Warmup khiến learning rate của các step đầu thấp hơn `1e-4`.

## Các notebook trên Colab

1. `01_hf_to_drive_rclone.ipynb`: chuẩn bị một bản dữ liệu dùng chung trên Drive (JSONL, ZIP ảnh, REVISION).
2. `02_sft_colab.ipynb`: train NF4, chọn best adapter local bằng val loss, kết thúc subprocess training, reload base BF16 rồi merge vào `best_model/`, upload model hoàn chỉnh; vẽ learning curves và chấm full validation một lần ở cuối.
3. `03_grpo_colab.ipynb`: recipe GRPO cũ, **giữ nguyên notebook**. Chưa chuyển sang giao diện model merged của experiment03.
4. `04_inference_colab.ipynb`: tải `experiment03/sft/best_model` từ Drive, load local BF16 bằng Transformers, sinh predictions devtest/test. Không tự tải lại Qwen gốc hay adapter riêng. Không yêu cầu adapter GRPO.

Mở notebook bản mới, restart runtime nếu vừa đổi dependency nhị phân và chạy từ đầu. Không sửa cấu hình để áp dụng vào một tiến trình training đang chạy: thay đổi chỉ có hiệu lực ở lượt chạy tiếp theo.

Setup mặc định clone repo tại `/content/semeval`, project tại `/content/semeval/task 6`. Kernel và subprocess dùng cùng `src`, cùng Python executable và project root. Các đường dẫn output/config/model trong notebook là tuyệt đối.

```text
/content/
├── rclone.conf
└── semeval/task 6/
    ├── src/mmcultureqa/
    ├── data/raw/
    ├── data/prepared/
    └── outputs/runs/experiment03/sft/
        ├── checkpoint-N/      chỉ local, phục vụ chọn best/resume
        ├── final/             adapter best local, phục vụ merge
        └── best_model/        model BF16 hoàn chỉnh, được upload
```

## Lưu trữ Drive

Dùng cùng `ROOT_FOLDER_ID`, `REMOTE`, `RCLONE_CONFIG_PATH` trong các notebook. Chỉ folder ID xác định root storage; không cần nhập đường dẫn MyDrive. Rclone dùng bản sao tạm của config, không sửa file gốc. `CHECKOUT` và `RCLONE_CONFIG_PATH` phải là đường dẫn tuyệt đối.

```text
MMCQA/                                ROOT_FOLDER_ID
├── task6-data/raw/                    3 JSONL English, 3 ZIP ảnh, REVISION
└── task6-runs/experiment03/
    ├── sft/
    │   ├── best_model/
    │   │   ├── model-*.safetensors    hoặc model.safetensors
    │   │   ├── model.safetensors.index.json  nếu chia shard
    │   │   ├── config.json
    │   │   ├── tokenizer/processor assets
    │   │   ├── prompt.txt
    │   │   ├── run_config.json        cấu hình/prompt/preprocessing đã đóng băng
    │   │   └── merge.json             base/revision/hash adapter/thông tin chọn best
    │   ├── selection.json
    │   ├── run.json
    │   ├── swift_arguments.json
    │   ├── logging.jsonl
    │   ├── learning_curves.png
    │   └── validation/
    │       ├── predictions.jsonl
    │       ├── scores.json
    │       └── evaluation.json
    └── predictions/
        ├── sft_devtest.jsonl
        └── sft_devtest/run.json
```

Danh sách upload được quy định trong `src/mmcultureqa/storage.py`. SFT mới dùng `merged_run`: chỉ upload model BF16 hoàn chỉnh và báo cáo cần thiết. **Không upload `final/` adapter, checkpoint trung gian, optimizer, dataset export, cache, ảnh giải nén hay file tạm merge.** Recipe GRPO cũ vẫn dùng bộ lọc adapter riêng, không đổi. File cũ trên Drive không bị tự xóa.

Model BF16 lớn hơn adapter, nên upload/download tốn dung lượng và thời gian hơn. Chỉ công bố thư mục `best_model` sau khi export đầy đủ; lượt merge lặp lại với cùng adapter được bỏ qua. Nếu adapter/base khác, exporter yêu cầu run mới, không ghi đè model hoàn chỉnh hiện có.

`final/` và `best_model/` không chứa optimizer/scheduler state. Resume chính xác một lượt train bị ngắt cần checkpoint đầy đủ đã tự sao lưu.

## Phạm vi LoRA và projector

Mặc định LoRA gắn vào **attention q/k/v/o** và **MLP gate/up/down** của các layer ngôn ngữ, không chỉ key/value. Base weights, embedding, lm_head và vision encoder giữ cố định.

Trong cell SFT:

```python
TRAIN_PROJECTOR = False
SFT_OVERRIDES = {}
```

Đổi `TRAIN_PROJECTOR=True` để thêm **LoRA** vào `model.visual.merger` và `model.visual.deepstack_merger_list.*` (các linear_fc1/linear_fc2). Vision encoder vẫn khóa; không full-finetune projector. Launcher kiểm tra thật sự có projector parameters trainable và ghi danh sách vào run metadata. Chỉ mở freeze_aligner mà không sửa target regex sẽ không đủ; code đã xử lý cả hai.

Không bảo đảm projector LoRA cải thiện kết quả. So sánh trên cùng split/seed. Nếu batch 8 không vừa VRAM:

```python
SFT_OVERRIDES = {
    'training.per_device_train_batch_size': 4,
    'training.gradient_accumulation_steps': 2,
    'training.per_device_eval_batch_size': 4,
}
```

Thay rank khi khởi tạo adapter mới, ví dụ `{'peft.r': 32, 'peft.lora_alpha': 64}`. Không coi việc đổi rank trên config là chuyển đổi một adapter đã train có kích thước khác.

## Train tiếp từ một model merged

SFT notebook có tùy chọn:

```python
# Cell Experiment settings:
RUN_NAME = 'experiment04'              # run mới
# Cell SFT:
BASE_MODEL_RUN_NAME = 'experiment03'   # tải best_model của run này làm base
```

Model merged cũ được tải từ Drive, lượng tử hóa NF4 khi load và gắn **adapter mới** để train. Kiến thức đã merge ở giai đoạn trước nằm trong base mới. Để trống BASE_MODEL_RUN_NAME thì dùng Qwen gốc ở revision đã pin.

Inference mặc định BF16, không lượng tử hóa lại bản merged. Processor, tokenizer, prompt và image preprocessing lấy từ artifact đã export. `local_files_only=True` khiến model loading báo lỗi nếu thiếu file, không tìm base trên Hub. Dataset và encoder mmBERT chấm điểm là nguồn riêng, vẫn cần tải/cache khi chưa có.

## Validation

`monitor.limit=256` chỉ giới hạn val định kỳ, không giới hạn train. Mẫu được chọn bằng random generator riêng seed 42, không lặp và không đổi qua các lần eval nếu manifest giữ nguyên. `null` dùng toàn bộ val; giới hạn lớn hơn số mẫu giữ nguyên thứ tự đầy đủ.

Trong lúc SFT mặc định không preload mmBERT, không sinh đáp án khi eval và chọn best bằng val loss. `PREDICT_WITH_GENERATE=True` là tùy chọn quay lại generation/F1, không phải default experiment03.

Cuối notebook, `VAL_LIMIT=None` sinh **một đáp án greedy cho mỗi câu trong đủ 1.000 mẫu** bằng chính model BF16 đã merge, rồi chấm scorer chính thức. `VAL_LIMIT=4` chỉ dành cho smoke evaluation và được lưu vào thư mục subset có tên riêng. Scorer cuối có BERTScore-F1/P/R, BLEU, ROUGE-L, Coverage. F1 không phải tỷ lệ trả lời đúng; token accuracy teacher forcing cũng không phải accuracy của đáp án tự sinh.

Best được chọn theo loss của model NF4 trên tập monitor. Chuyển sang base BF16 có thể làm thay đổi điểm/đáp án; best_model là artifact của checkpoint được chọn, không phải chứng nhận BF16 tốt nhất trong mọi checkpoint. Final BF16 score được lưu để đối chiếu.

## CLI và kiểm tra

Python 3.11+; GPU hỗ trợ BF16 cho training và inference mặc định. Không đổi dependency organizer chồng lên môi trường GPU.

```bash
pip install -r requirements.txt
pip install --no-deps -e .
python -m mmcultureqa.data.prepare download
python -m mmcultureqa.data.prepare prepare
python -m mmcultureqa.training.sft --config configs/sft.yaml
python -m mmcultureqa.export --run-dir outputs/runs/experiment03/sft
python -m mmcultureqa.evaluate --run-dir outputs/runs/experiment03/sft
python -m mmcultureqa.inference.predict --config configs/inference.yaml
```

Export chạy riêng sau training để giải phóng model NF4 trước khi reload BF16. `--device cpu` cho merge là lựa chọn thủ công nếu có đủ RAM; mặc định cuda:0. Checkpoint quantized không được dùng trực tiếp làm đối tượng merge.

```bash
python tools/build_notebooks.py
# Từ repo root, sau khi cài PyYAML:
python -m unittest discover -s "task 6/tests" -v
```

vLLM là backend tùy chọn riêng qua `requirements-vllm.txt`; experiment03 không dùng. ms-swift vẫn cần `msgspec` cho import GRPO utilities nên dependency được khai báo trực tiếp. Preflight kiểm tra concrete trainer imports trước khi load model.

Các kiểm tra CPU dùng fixture/mock để xác minh cấu hình, đường dẫn Colab, random subset, projector regex, lựa chọn adapter merge, tải model local-only, upload allowlist và quy trình BF16 evaluation. Chúng **không chứng minh** GPU merge/generation/backward thực tế chạy đúng, batch 8 vừa VRAM hay tốc độ đạt mục tiêu. Xem `docs/VALIDATION.md`.

## Dữ liệu và nguồn

Snapshot dataset `QCRI/MMCQA-SemEval27`, revision `e850ede73e2f1ef3dde12e0210de67d107247ef0`: English textual QA của MENA, train 10.000, dev 1.000, devtest 1.000 không nhãn. Train được chia theo nhóm ảnh, seed 42 thành 9.000 train / 1.000 validation. Dev/devtest giữ riêng. Manifest ảnh tương đối với project root, không phụ thuộc đường dẫn máy phát triển.

Preparation kiểm tra duplicate/overlap ID và ảnh, ZIP CRC và khả năng giải mã ảnh. Scorer organizer trong `vendor/organizer` giữ nguyên. JSONL và Parquet là cùng dữ liệu; backup Drive chỉ giữ JSONL cần thiết cùng ZIP ảnh. Không dùng val/test để tạo dữ liệu training hay chọn prompt dựa trên test answers.

Thông tin chi tiết: `docs/DATASET.md`, `docs/DATA_INSPECTION.json`, `docs/SOURCES.md`. Thay đổi này không sửa Task 8.
