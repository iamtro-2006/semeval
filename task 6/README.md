# Task 6 · MMCultureQA English textual QA

Pipeline riêng trong `D:\semeval\task 6`, không thay đổi Task 8. SFT dùng Transformers Trainer + PEFT; GRPO dùng TRL + PEFT. Không dùng ms-swift.

## Bắt đầu trên Colab

1. Mở `notebooks/01_hf_to_drive_rclone.ipynb` trên Colab CPU. Cell đầu clone `https://github.com/iamtro-2006/semeval.git`, checkout `GIT_REF`, vào `task 6/` và cài package. Notebook tải English QA trực tiếp từ Hugging Face rồi dùng rclone copy archive và nhãn lên Google Drive. Trong cell cấu hình, sửa `RCLONE_CONFIG_PATH`, `REMOTE`, `ROOT_FOLDER_ID` và `DESTINATION_SUBDIR`. Notebook dùng bản sao tạm của config, không sửa file gốc.
2. Mở `notebooks/02_sft_grpo_inference_colab.ipynb` trên Colab GPU. Cell đầu cũng clone repo này. Notebook copy archive từ Drive sang SSD Colab, giải nén, prepare dữ liệu, chạy SFT → GRPO → inference → scorer.
3. Để `RUN_FULL=False` cho lượt thử 1 optimizer step mỗi phương pháp. Sau khi xác nhận chạy được, dùng tên experiment mới và `RUN_FULL=True`. Smoke không phải kết quả thực nghiệm.
4. Checkpoint lưu vào Drive; ảnh và xử lý dữ liệu ở SSD Colab. Các cell chạy trong subprocess nên GPU được giải phóng giữa các stage. Khi Colab yêu cầu restart sau cài package, restart rồi chạy lại setup.

Notebook lấy code trực tiếp từ GitHub; không đóng gói hay upload ZIP code. Sửa `REPO_URL` và `GIT_REF` trong notebook để chọn repo/branch/commit. Khi chỉnh generator, sinh lại notebook:

```powershell
python tools/build_notebooks.py
```

Ví dụ cell rclone khi thư mục root là `MMCQA` trong My Drive:

```python
RCLONE_CONFIG_PATH = '/content/drive/MyDrive/configs/rclone.conf'
REMOTE = 'gdrive'
ROOT_FOLDER_ID = 'ID_THU_MUC_MMCQA'  # cũng nhận URL folder đầy đủ
DESTINATION_SUBDIR = 'task6-data/raw'
TRANSFERS = 4
CHECKERS = 8
```

Nếu file config nằm trên Drive, chạy cell mount Drive tùy chọn trước. Nếu upload config qua Files panel, dùng `/content/rclone.conf`. Để root rỗng và subdir rỗng sẽ dùng `My Drive/MMCQA/task6-data/raw`. Nếu đổi thư mục đích khác cấu trúc mặc định, sửa `drive_raw` tương ứng trong notebook thực nghiệm; folder ID của rclone không tự thay đổi đường dẫn Drive mount.

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
