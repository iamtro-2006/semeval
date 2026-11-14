# Task 6 · MMCultureQA English textual QA

Pipeline riêng trong `D:\semeval\task 6`, không thay đổi Task 8. SFT dùng Transformers Trainer + PEFT; GRPO dùng TRL + PEFT. Không dùng ms-swift.

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
    │   ├── checkpoint-N/
    │   └── final/
    ├── grpo/                    hoặc grpo-smoke
    │   ├── checkpoint-N/
    │   └── final/
    └── predictions/             JSONL và scores
```

1. `01_hf_to_drive_rclone.ipynb`: tải từ Hugging Face trên Colab, upload nhãn/ZIP vào `task6-data/raw` bên dưới root.
2. `02_sft_colab.ipynb`: train SFT, upload checkpoint/adapter, sau đó **inference toàn bộ validation và chấm scorer chính thức**, lưu `validation/scores.json` và predictions lên Drive.
3. `03_grpo_colab.ipynb`: chỉ train GRPO tiếp từ SFT; chọn `SFT_RUN_NAME`, `INPUT_SFT_STAGE`, `SFT_CHECKPOINT`. Cuối cùng **inference toàn bộ validation và chấm scorer chính thức**, lưu kết quả riêng bên trong run GRPO.
4. `04_inference_colab.ipynb`: chỉ inference test (`devtest` mặc định), tự upload predictions. Nếu không có nhãn công khai thì chỉ xuất file nộp, không tính điểm local. Không dùng notebook này để chấm validation.

Mỗi notebook clone repo `iamtro-2006/semeval` và cài package trong `task 6/`. Không ZIP/upload code. Dùng cùng `ROOT_FOLDER_ID` trong các phiên để đồng bộ. File rclone gốc không bị sửa; dùng bản sao tạm để refresh token. Nếu config nằm trên Drive thì chạy cell mount tùy chọn chỉ để đọc config.

`RUN_FULL=False` mặc định chạy thử một optimizer step; full run dùng tên experiment mới và `RUN_FULL=True`. Khi chạy GRPO đầy đủ, chọn adapter SFT đầy đủ. `final` chứa adapter + processor, không chứa bản sao toàn bộ Qwen. Resume cùng stage dùng `RESUME_CHECKPOINT='checkpoint-N'`, giữ nguyên cấu hình; chuyển SFT sang GRPO dùng adapter và optimizer GRPO mới.

Đánh giá cuối trong hai notebook train mặc định luôn dùng **đủ 1.000 mẫu validation**, độc lập với số mẫu/step smoke training. `VAL_LIMIT=None` là full; đổi thành `4` chỉ để thử pipeline đánh giá, kết quả subset được lưu tên riêng, không gọi là full-val. Scorer báo BERTScore-F1/P/R, BLEU, ROUGE-L và Coverage; đây không phải tỷ lệ exact-match accuracy. Đánh giá cuối không thay thế monitoring định kỳ: SFT full run vẫn chọn checkpoint bằng eval_loss, GRPO vẫn chưa chọn checkpoint bằng điểm val.

Chạy lại đánh giá adapter cuối từ CLI, không train lại:

```bash
python -m mmcultureqa.evaluate --run-dir outputs/sft
python -m mmcultureqa.evaluate --run-dir outputs/grpo
```

Evaluator dùng prompt/model settings đã lưu trong `run.json`, kiểm tra manifest không đổi và train/val không trùng ID/ảnh, sinh đáp án rồi chấm scorer trong hai subprocess để giải phóng VRAM giữa VLM và mmBERT. Chỉ đọc gold để chấm; gold không đưa vào prompt.

Training và ảnh dùng SSD local; rclone copy checkpoint theo lịch save và cuối stage. Nếu Colab bị ngắt đột ngột, chỉ checkpoint đã upload xong mới phục hồi được. Remote dùng `copy`, không xóa checkpoint cũ trên Drive; save_total_limit chỉ giới hạn bản local. CLI ngoài notebook vẫn hoạt động bình thường, không sync nếu không cấu hình môi trường callback.

Sửa notebook generator rồi sinh lại:

```bash
python tools/build_notebooks.py
```

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
    │   ├── training/          sft.py, grpo.py, rewards.py
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
pip install --no-deps -e .
python -m mmcultureqa.data.prepare download
python -m mmcultureqa.data.prepare prepare
python -m mmcultureqa.training.sft --config configs/sft.yaml
python -m mmcultureqa.training.grpo --config configs/grpo.yaml
python -m mmcultureqa.inference.predict --config configs/inference.yaml
python -m mmcultureqa.inference.predict --config configs/inference.yaml --set model.adapter=outputs/grpo/final --set inference.output=outputs/predictions/grpo_dev.jsonl
python -m mmcultureqa.inference.predict --config configs/inference.yaml --set model.adapter=outputs/grpo/final --set inference.split=devtest --set inference.output=outputs/predictions/grpo_devtest.jsonl
```

Ví dụ chỉnh tham số, prompt và resume:

```bash
python -m mmcultureqa.training.sft --config configs/sft.yaml --set model.load_in_4bit=true --set training.learning_rate=0.0001 --set prompt_file=prompts/qa.txt
python -m mmcultureqa.training.grpo --config configs/grpo.yaml --set model.load_in_4bit=true --set reward.device=cuda
python -m mmcultureqa.training.sft --config configs/sft.yaml --set resume_from_checkpoint=outputs/sft/checkpoint-100
```

Full precision là default ở YAML. Notebook bật 4-bit một cách tường minh để thử trên GPU ít VRAM; không tự động lượng tử hóa, giảm pixel hay cắt chuỗi khi OOM. Dtype auto chọn BF16 nếu GPU hỗ trợ, nếu không FP16. Số generation, batch và gradient accumulation phải phù hợp quy tắc chia nhóm của GRPO; default 1 GPU: batch 1 × accumulation 4, group 4. Không bảo đảm mọi Colab GPU đều đủ VRAM cho mọi cấu hình.

## Điều kiện huấn luyện

- SFT dùng native chat template, processor gốc và ảnh; chỉ tính loss trên đáp án và token kết thúc. Prompt, image tokens và padding không tính loss. Có kiểm tra prefix mask với mẫu thật trước train. Không packing hay truncation ảnh/đáp án.
- LoRA mới mặc định chỉ gắn vào attention/MLP của language model, thể hiện rõ bằng regex. Muốn thay phạm vi thì sửa `peft.target_modules`; danh sách tham số thực sự học được ghi vào `run.json`. Đây không phải thiết lập train toàn bộ vision encoder.
- GRPO nạp adapter SFT với `is_trainable=True` và cập nhật tiếp adapter đó; không tạo adapter ngẫu nhiên thay thế. Tham số `peft` chỉ dùng khi khởi tạo LoRA mới. Khi nạp adapter cũ, rank/target modules lấy từ adapter cũ.
- GRPO sinh rollout mới từ train; nhãn `answer` chỉ đưa cho reward, không đưa vào prompt. `max_prompt_length=null` để tránh cắt image tokens.
- Recipe continuation đặt `beta=0`. TRL 0.24 mặc định vô hiệu hóa adapter để làm KL reference, tức base trước SFT. Script từ chối beta khác 0 để không âm thầm dùng reference sai. Muốn KL với frozen SFT cần bổ sung reference riêng.
- Reward mmBERT tải một lần, chạy batch, pin revision/layer theo scorer. Đây là semantic reward, không phải verifier tuyệt đối; hàm reward nằm riêng để bạn chỉnh. `reward.type=exact` chỉ phù hợp kiểm tra hoặc dữ liệu có đáp án khớp chính xác. CPU reward tiết kiệm VRAM nhưng có thể là nút thắt tốc độ; chọn CUDA khi đủ bộ nhớ.
- SFT `final` mặc định là checkpoint có eval_loss tốt nhất. GRPO `final` là checkpoint cuối; script không tự tuyên bố là checkpoint tốt nhất. Checkpoint định kỳ giữ optimizer/scheduler cho resume; `final` chỉ dùng adapter continuation/inference.
- Không nuốt lỗi ảnh, lỗi generate hoặc OOM. Inference lưu JSONL tăng dần và kiểm tra fingerprint trước resume; đổi model/prompt/split thì dùng output mới. Không sửa adapter tại cùng đường dẫn rồi resume prediction cũ.

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

Đã tải và kiểm tra đủ 12.000 ảnh local, đủ ba JSONL và ba Parquet. Đã kiểm tra cú pháp code/notebook, manifest và cấu hình trên CPU. Máy phát triển không có GPU: chưa chạy forward/backward Qwen3-VL, chưa xác nhận loss/gradient trên GPU, chưa chạy thực nghiệm Colab và chưa có kết quả chứng minh GRPO cải thiện. Notebook cung cấp lượt thử GPU để xác nhận trước full run.

Tham khảo API theo phiên bản được pin: `docs/SOURCES.md`.
