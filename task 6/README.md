# Task 6 · MMCultureQA · SFT và reasoning distillation

Pipeline dùng Qwen3-VL-4B-Instruct: train QLoRA 4-bit, chọn checkpoint bằng validation loss, xuất **một model BF16 hoàn chỉnh**, rồi sinh đánh giá cuối/inference bằng **vLLM** sau khi tiến trình training đã thoát. Có hai hướng độc lập: SFT đáp án `experiment03` và SFT reasoning từ teacher `experiment04_reasoning`. Notebook GRPO giữ nguyên recipe và tên run cũ.

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
| Tập monitor SFT | Toàn bộ 1.000 mẫu validation (`monitor.limit=null`) |
| Chọn best | Val loss thấp nhất |
| LoRA mặc định trong YAML | r=16, alpha=32; chỉnh qua SFT_OVERRIDES nếu muốn |
| Projector LoRA | Tùy chọn, mặc định tắt |
| Artifact trên Drive | `best_model/`: model BF16 đã merge + processor/tokenizer |

9.000 câu train × 2 epoch / batch 8 = 2.250 optimizer step, khoảng 10 kỳ validation. Đây không phải dự đoán thời gian GPU. Warmup khiến learning rate của các step đầu thấp hơn `1e-4`.

## Các notebook trên Colab

1. `01_hf_to_drive_rclone.ipynb`: chuẩn bị một bản dữ liệu dùng chung trên Drive (JSONL, ZIP ảnh, REVISION).
2. `02_sft_colab.ipynb`: train NF4, chọn best adapter local bằng val loss, kết thúc subprocess training, reload base BF16 rồi merge vào `best_model/`, upload model hoàn chỉnh; vẽ learning curves, chấm full validation trực tiếp rồi chấm lại cùng checkpoint bằng zero-shot CoT; có cell bảng so sánh metrics.
3. `03_grpo_colab.ipynb`: recipe GRPO cũ, **giữ nguyên notebook**. Chưa chuyển sang giao diện model merged của experiment03.
4. `04_inference_colab.ipynb`: tải `best_model` từ Drive, load local BF16 bằng vLLM, sinh predictions devtest/test. Mặc định `experiment03`; để chạy student mới, đặt `RUN_NAME` và `INPUT_RUN_NAME` thành `experiment04_reasoning`. Tự phục hồi prompt/định dạng reasoning từ artifact, chỉ lấy đáp án cuối để chấm/nộp.
5. `05_teacher_reasoning_colab.ipynb`: dùng Qwen3-VL-8B-Thinking-FP8 chính thức qua vLLM, nhìn ảnh + câu hỏi + đáp án chuẩn, tạo các bước suy luận văn hóa cho **9.000 mẫu train**, upload dữ liệu lên Drive. Không train student ở đây.
6. `06_sft_reasoning_colab.ipynb`: tải dữ liệu teacher hoàn chỉnh, kiểm tra hash/split/coverage, SFT Qwen3-VL-4B-Instruct với reasoning và đáp án; merge best BF16 và chấm full validation bằng vLLM. Run riêng `experiment04_reasoning`.
7. `07_base_vs_sft_cot_colab.ipynb`: **một notebook inference cho cả hai model**: Qwen3-VL-4B-Instruct gốc từ Hub và SFT best từ Drive. Chạy tuần tự bằng vLLM ở FP16, cùng user prompt zero-shot CoT ngắn. Giữ nguyên system prompt cũ (base từ config, SFT từ artifact); không train lại hoặc sửa bản model BF16 trên Drive. Có bảng metrics base/SFT/Δ và raw reasoning để kiểm tra.

Notebook 07 mặc định dùng `SFT_RUN_NAME='experiment03'`, `SFT_MODEL_STAGE='sft'`, full validation 1.000 câu; `LIMIT=4` để thử nhanh. User prompt nằm trong `prompts/qa_cot_simple.txt`: chỉ yêu cầu suy luận từng bước từ ảnh và kiến thức văn hóa, xuất reasoning/answer bằng hai cặp thẻ, không có checklist bốn bước. System prompt không chứa thêm yêu cầu CoT.

Cả hai model inference FP16 không quantize, cùng image limits, seed 42, temperature 0.7, top-p 0.8, top-k 20 và presence penalty 1.5. Mẫu sai format retry tối đa một lần; vẫn sai thì dừng và giữ trace. Chạy một model rồi thoát subprocess, chấm điểm trong subprocess riêng, sau đó mới chạy model tiếp theo. Kết quả mới lưu `MMCQA/task6-inference/base_fp16_vs_sft_cot01/`, gồm predictions/traces/config/scores của base và SFT, bảng HTML/CSV/JSON; không upload lại model weights. Với devtest/test chưa có nhãn, chỉ xuất predictions/traces.

Nếu chạy hướng reasoning, thứ tự là **01 (nếu chưa có dữ liệu) → 05 → 06 → 04**. Notebook 02 là baseline riêng, không phải điều kiện để chạy 06. Chi tiết prompt, cấu hình và đường dẫn Drive: [docs/REASONING.md](docs/REASONING.md).

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

Notebook 02 chạy hai lượt cuối trên **cùng model BF16 và cùng 1.000 câu validation**: zero-shot trả lời trực tiếp (`validation/`), sau đó zero-shot CoT (`validation_cot/`). CoT dùng đúng prompt 4 bước trong `prompts/qa_cot.txt`, gồm lưu ý loại watermark text. Không đổi prompt SFT hay checkpoint đã merge. Cả hai mặc định greedy; token budget direct lấy từ monitor (256), CoT mặc định 1.024 (`COT_MAX_NEW_TOKENS`), context 4.096. Khác biệt budget được ghi trong báo cáo để diễn giải so sánh.

Cell cuối hiển thị bảng BERTScore-F1/P/R, BLEU, ROUGE-L, Coverage và Δ (CoT − direct). BERTScore/ROUGE/Coverage nhân 100, BLEU giữ nguyên; Δ theo điểm, không phải tăng trưởng phần trăm hay kiểm định thống kê. Trước khi lập bảng, kiểm tra cùng checkpoint signature, gold hash, số mẫu, seed, backend và precision scorer. Zero-shot ở đây nghĩa là không có in-context demonstration, vẫn dùng checkpoint đã SFT.

Chỉ nội dung `<answer>` của CoT được chấm. Nếu format thiếu/sai, lưu raw generation và `format_error`, retry đúng mẫu đó **tối đa 1 lần** với sampling temperature 0.7, seed +1. Nếu vẫn sai thì báo lỗi và dừng, không ghi đáp án rỗng hay bỏ mẫu. Predictions đã hoàn tất vẫn được giữ để resume. Lượt sinh chính của hai chế độ vẫn greedy; riêng retry CoT có sampling và được ghi rõ trong trace.

Progress inference/validation dùng tqdm (phần trăm, thanh tiến độ, ETA); BERTScore bật progress qua wrapper, không sửa scorer organizer. Các file `prompt_comparison.html`, `.csv`, `.json`, kết quả `validation_cot/` và prompt CoT được upload Drive cạnh kết quả direct. Nếu đã train xong, chỉ cần dùng repo mới và chạy lại các cell validation/report; không cần train lại. Rerun direct có thể dùng predictions cùng signature đã hoàn tất, rồi tạo thêm metadata cần cho bảng.

SFT mặc định `monitor.limit=null`, dùng đủ 1.000 mẫu val cho loss/token accuracy định kỳ, không giới hạn train. Nếu chủ động đặt một giới hạn nhỏ hơn số mẫu, subset được chọn ngẫu nhiên seed 42, không lặp và không đổi qua các lần eval nếu manifest giữ nguyên. Recipe GRPO đang tạm dừng giữ riêng `monitor.limit=256` như trước, không bị thay đổi theo mặc định SFT.

Trong lúc SFT mặc định không preload mmBERT, không sinh đáp án khi eval và chọn best bằng val loss. `PREDICT_WITH_GENERATE=True` là tùy chọn quay lại generation/F1, không phải default experiment03.

Cuối notebook, `VAL_LIMIT=None` sinh **một đáp án greedy cho mỗi câu trong đủ 1.000 mẫu** bằng chính model BF16 đã merge, rồi chấm scorer chính thức. `VAL_LIMIT=4` chỉ dành cho smoke evaluation và được lưu vào thư mục subset có tên riêng. Scorer cuối có BERTScore-F1/P/R, BLEU, ROUGE-L, Coverage. F1 không phải tỷ lệ trả lời đúng; token accuracy teacher forcing cũng không phải accuracy của đáp án tự sinh.

Best được chọn theo loss của model NF4 trên tập monitor. Chuyển sang base BF16 có thể làm thay đổi điểm/đáp án; best_model là artifact của checkpoint được chọn, không phải chứng nhận BF16 tốt nhất trong mọi checkpoint. Final BF16 score được lưu để đối chiếu.

## CLI và kiểm tra

Python 3.11+; GPU hỗ trợ BF16 cho training và inference mặc định. Không đổi dependency organizer chồng lên môi trường GPU.

```bash
pip install -r requirements-vllm.txt
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

Notebook 02/04/05/06 cài `requirements-vllm.txt` mặc định. Generation cuối/inference chạy vLLM; validation định kỳ của SFT vẫn teacher forcing. Notebook 03 giữ backend Transformers và recipe cũ. ms-swift vẫn cần `msgspec` cho import GRPO utilities nên dependency được khai báo trực tiếp. Preflight kiểm tra concrete trainer imports trước khi load model.

Các kiểm tra CPU dùng fixture/mock để xác minh cấu hình, đường dẫn Colab, random subset, projector regex, lựa chọn adapter merge, tải model local-only, upload allowlist và quy trình BF16 evaluation. Chúng **không chứng minh** GPU merge/generation/backward thực tế chạy đúng, batch 8 vừa VRAM hay tốc độ đạt mục tiêu. Xem `docs/VALIDATION.md`.

## Dữ liệu và nguồn

Snapshot dataset `QCRI/MMCQA-SemEval27`, revision `e850ede73e2f1ef3dde12e0210de67d107247ef0`: English textual QA của MENA, train 10.000, dev 1.000, devtest 1.000 không nhãn. Train được chia theo nhóm ảnh, seed 42 thành 9.000 train / 1.000 validation. Dev/devtest giữ riêng. Manifest ảnh tương đối với project root, không phụ thuộc đường dẫn máy phát triển.

Preparation kiểm tra duplicate/overlap ID và ảnh, ZIP CRC và khả năng giải mã ảnh. Scorer organizer trong `vendor/organizer` giữ nguyên. JSONL và Parquet là cùng dữ liệu; backup Drive chỉ giữ JSONL cần thiết cùng ZIP ảnh. Không dùng val/test để tạo dữ liệu training hay chọn prompt dựa trên test answers.

Thông tin chi tiết: `docs/DATASET.md`, `docs/DATA_INSPECTION.json`, `docs/SOURCES.md`. Thay đổi này không sửa Task 8.
