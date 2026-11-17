# Task 6 · MMCultureQA · Answer-only SFT

Pipeline hiện tại: QLoRA NF4 4-bit + BF16 compute → chọn checkpoint bằng val loss → merge adapter đã chọn vào base BF16 → đánh giá cuối/inference bằng vLLM. Không còn lượt validation CoT trong notebook SFT.

## Notebook đang dùng

| Notebook | Chức năng | Run mặc định |
|---|---|---|
| 01_hf_to_drive_rclone | Chuẩn bị dữ liệu chung trên Drive | task6-data/raw |
| 02_sft_colab | SFT Qwen3-VL-4B-Instruct | experiment03 |
| 03_grpo_colab | Recipe GRPO cũ đang tạm dừng | experiment01_qlora |
| 04_inference_colab | Tải model merged của một trong ba model và inference | chọn INPUT_RUN_NAME |
| 05_sft_qwen3vl8b_colab | SFT Qwen3-VL-8B-Instruct | experiment05_qwen3vl8b |
| 06_sft_qwen35_4b_colab | SFT Qwen3.5-4B (bản post-trained) | experiment06_qwen35_4b |

Đã xóa notebook teacher reasoning, reasoning SFT và base-vs-SFT CoT cũ (05/06/07). Các báo cáo/dữ liệu cũ trên Drive không bị thay đổi. Một số module/CLI và test về artifact cũ được giữ để đọc lại kết quả trước đây; chúng không chạy trong notebook SFT mới.

## Cấu hình training

| Tham số | Qwen3-VL-4B | Qwen3-VL-8B | Qwen3.5-4B |
|---|---:|---:|---:|
| Microbatch | 8 | 2 | 4 |
| Gradient accumulation | 1 | 4 | 2 |
| Effective batch, 1 GPU | 8 | 8 | 8 |
| Eval batch | 8 | 2 | 4 |
| Epoch mặc định | 2 | 2 | 2 |
| Peak LR | 1e-4 | 1e-4 | 1e-4 |
| Eval/save interval | 225 | 225 | 225 |

Giữ cấu hình epoch hiện tại; nếu muốn thử 1 epoch, điền `SFT_OVERRIDES = {'training.num_train_epochs': 1}` trong cell SFT. Warmup 3%, cosine decay. Tất cả train NF4 + double quantization, BF16 compute; base frozen, LoRA rank 16/alpha 32 mặc định. Cấu hình chưa benchmark GPU về tốc độ hay VRAM.

**Val loss được đo ngay step 0, trước cập nhật trọng số đầu tiên**, bằng `training.eval_on_start=true`; sau đó tiếp tục mỗi 225 step. Dùng đủ 1.000 mẫu validation cho teacher-forced loss/token accuracy, không generation/mmBERT trong training. Đây là giá trị đo thực để nối graph, không nội suy hoặc chèn điểm giả. Điểm step 0 không phải checkpoint được save; best checkpoint vẫn được chọn trong lịch eval/save. Resume có thể đánh giá tại bước đang resume.

LoRA Qwen3-VL gồm attention q/k/v/o và MLP gate/up/down. Qwen3.5 thêm các projection của linear attention (in_proj_qkv/z/b/a và out_proj), cùng full attention/MLP. `TRAIN_PROJECTOR=True` tùy chọn thêm LoRA vào merger; vision encoder vẫn frozen. Launcher xác nhận thực sự có tham số projector trainable khi bật.

Qwen3.5 là hybrid-thinking VLM, nhưng lượt SFT này **answer-only**, dùng enable_thinking=false và non-thinking prefix theo template của framework. Inference khôi phục cùng lựa chọn từ model artifact. Không thêm prompt CoT.

## Hai môi trường GPU

- **Qwen3-VL-4B/8B:** `requirements-vllm.txt`, ms-swift 3.11.3, Transformers 4.57.6, torch 2.8.0, vLLM 0.11.0.
- **Qwen3.5-4B:** `requirements-qwen35.txt` riêng, ms-swift 4.1.3, Transformers 5.6.2, torch 2.11.0, vLLM 0.20.0. Bộ cũ không có kiến trúc qwen3_5. Không cài chồng cả hai requirements cùng lúc.

Notebook 06 tự chọn môi trường Qwen3.5. Trong notebook 04, chọn `RUNTIME_PROFILE='qwen35'` ở cell cài đặt trước khi tải Qwen3.5; với Qwen3-VL chọn `'legacy'`. Dùng runtime sạch khi chuyển họ model; restart nếu vừa thay thư viện nhị phân. Các API Swift 4.x được xử lý riêng, không sửa recipe GRPO cũ.

Các pin được đối chiếu metadata package và source API chính thức; chưa thực thi load/training/vLLM trên GPU. Qwen3.5 linear attention có thể dùng đường fallback khi thiếu kernel tối ưu; hiệu năng phải đo trên Colab thực tế.

## Lưu model và inference

Sau training: lưu adapter đã chọn vào final/ local → tiến trình train thoát → reload base BF16 riêng → safe merge → best_model/ đầy đủ → upload Drive. Không upload adapter riêng, intermediate checkpoints, optimizer, cache hay ảnh giải nén. Tất cả dùng đường dẫn project tuyệt đối dưới `/content/semeval/task 6`.

```text
MMCQA/
  task6-data/raw/
  task6-runs/<RUN_NAME>/sft/
    best_model/         weights BF16 + processor/tokenizer + prompt/run_config/merge metadata
    selection.json
    run.json
    logging.jsonl
    learning_curves.png
    validation/         predictions + điểm cuối
```

Notebook 04: sửa `INPUT_RUN_NAME` thành một trong ba run trong bảng, `MODEL_STAGE='sft'`, chọn runtime profile tương ứng. Model type, system/user prompt và image preprocessing lấy từ artifact đã lưu; không gắn nhầm adapter hoặc tải base gốc thay bản SFT. Qwen3.5 được kiểm tra runtime profile trước khi inference. Các model được pin revision trong YAML.

Full validation cuối vẫn sinh câu trả lời cho đủ 1.000 mẫu bằng vLLM BF16, rồi chấm bằng scorer chính thức: BERTScore P/R/F1, BLEU, ROUGE-L, Coverage. VLM và scorer chạy ở subprocess riêng để giải phóng VRAM. Progress bar được giữ cho inference và BERTScore. `VAL_LIMIT=4` chỉ là tùy chọn smoke evaluation; smoke training không tự giới hạn full validation cuối.

## Kiểm tra

Từ repo root với PyYAML/tqdm đã cài:

```bash
python "task 6/tools/build_notebooks.py"
python -m unittest discover -s "task 6/tests" -v
```

CPU tests dùng fixture/mock để kiểm tra đường dẫn/import Colab, model-family dispatch, argument mapping Swift 3/4, NF4/BF16, target LoRA linear attention, step-0 curve data, metadata artifact, inference và upload filters. Chúng không thay thế kiểm tra GPU, không chứng minh cải thiện điểm hay đảm bảo vừa VRAM.

Dữ liệu QCRI/MMCQA-SemEval27 dùng revision e850ede73e2f1ef3dde12e0210de67d107247ef0, split theo nhóm ảnh seed 42: 9.000 train / 1.000 validation; dev/devtest giữ riêng. Organizer scorer không thay đổi.

Nguồn Qwen3.5: https://swift.readthedocs.io/en/v4.1/BestPractices/Qwen3_5-Best-Practice.html và https://huggingface.co/Qwen/Qwen3.5-4B.
