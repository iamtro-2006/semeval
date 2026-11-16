# Cultural reasoning distillation

Thứ tự: **01 (nếu chưa có dữ liệu) → 05 teacher → 06 SFT student → 04 inference**. SFT đáp án notebook 02 là baseline riêng.

## Smoke trên 100 mẫu ngẫu nhiên

- Notebook 05: đặt `SMOKE=True`. Teacher chọn **100 câu train ngẫu nhiên, không lặp, seed 42**; ghi vào `task6-data/reasoning/cultural_teacher01-smoke100/`. Full run vẫn dùng `cultural_teacher01/`.
- Notebook 06: đặt `RUN_FULL=False` trong cell Experiment settings trước khi chạy các cell tiếp theo. Tự tải teacher `-smoke100`, train trên tối đa 100 mẫu accepted, lấy ngẫu nhiên seed 42; output riêng `task6-runs/experiment04_reasoning/sft-smoke/`. Nếu muốn lấy 100 mẫu từ teacher đầy đủ, đặt thêm `TEACHER_SMOKE_DATA=False` trong cell tải dữ liệu.
- Smoke student chỉ giảm dữ liệu train; giữ nguyên số epoch trong YAML (mặc định 2), lịch validation/chọn best và **full validation 1.000 mẫu ở cuối bằng vLLM**. Với ít bước train, có thể chưa tới mốc eval định kỳ 225 bước; không ép thành một optimizer step. Adapter cuối vẫn được merge BF16 và upload vào stage smoke riêng. Không ghi đè stage `sft`.
- Nếu teacher loại một số mẫu trong 100 câu đã chọn, student chỉ học các mẫu hợp lệ còn lại; ngưỡng accepted mặc định vẫn ≥95% tập teacher đã chọn. Không dùng mẫu lỗi để bù đủ 100. Full training tiếp tục từ chối dữ liệu teacher smoke.
- Dataset lưu danh sách ID/seed; kiểm tra lại selection, hash và split khi tải. Việc lấy mẫu dùng RNG riêng, không thay đổi RNG toàn cục hay lấy 100 dòng đầu tiên.

## Teacher

Mặc định [Qwen/Qwen3-VL-8B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-FP8), bản **FP8 đã quantize do Qwen phát hành**, không tự quantize BF16 lúc load. Recipe block-FP8 này yêu cầu L4/Ada, Hopper hoặc mới hơn; không dùng T4. Teacher và student chạy ở các notebook/tiến trình riêng, không cùng chiếm GPU. Revision teacher được resolve thành commit bất biến ở lần đầu và giữ nguyên khi resume.

Prompt nằm trong `prompts/teacher_reasoning.txt`: 3–6 bước, khoảng tối đa 300 từ, gồm kiến thức cần dùng, bằng chứng và vị trí trong ảnh, chữ/ngôn ngữ khi đọc được, đối chiếu kiến thức văn hóa, rồi suy ra đáp án. Chỉ dùng dấu hiệu liên quan; không tự gán vùng hay bịa chi tiết để khớp đáp án. Teacher nghĩ trước rồi xuất JSON chứa các bước đã tổ chức. Thinking gốc được giữ trong journal để kiểm tra; target student dùng chuỗi bước ngắn gọn, không sao chép nguyên phần tự thoại dài.

Đây là **lời giải tổng hợp có điều kiện theo đáp án chuẩn**. Kiểm tra format/finish reason và teacher tự báo `supported` không chứng minh văn hóa hay chi tiết ảnh là đúng. Xem `audit_sample.jsonl` (ngẫu nhiên seed 42) và `rejected.jsonl` trước khi train. Mẫu lỗi, cắt token hoặc thiếu bằng chứng thử lại tối đa 2 lần rồi bị loại, không gán reasoning giả để đủ số lượng.

Mặc định teacher: batch gửi 8, tối đa 4 sequence đồng thời, context 8.192, tối đa 4.096 token sinh/mẫu, GPU memory utilization 0,85. Đây là điểm khởi đầu chưa benchmark. Chỉnh qua `TEACHER_OVERRIDES`; chọn tên run mới khi đổi model/prompt/cấu hình để không trộn dữ liệu. `SMOKE=True` tự đặt giới hạn 100 và tên riêng; dữ liệu smoke không được dùng cho full SFT.

## Dữ liệu trên Drive

`MMCQA/task6-data/reasoning/cultural_teacher01/`:

- `train_reasoning.jsonl`: mẫu accepted, gồm ID/ảnh/câu hỏi/đáp án gốc và reasoning.
- `dataset.json`: trạng thái hoàn tất, số accepted/rejected, hash train/validation/dữ liệu đầu ra, teacher revision.
- `generation.json`: prompt/cấu hình và revision đã khóa.
- `attempts.jsonl`: journal để resume/audit, có output gốc và lỗi từng lần thử.
- `rejected.jsonl`, `audit_sample.jsonl`: mẫu loại và mẫu kiểm tra.

Chỉ lưu **một bộ ảnh chung** ở `task6-data/raw`; không upload teacher weights hay copy ảnh theo mẫu. Notebook upload journal khi subprocess kết thúc, kể cả kết thúc với lỗi. Nếu Colab mất runtime trước bước upload, journal chỉ local sẽ mất; để tiếp tục journal đã upload, đặt `RESUME_FROM_DRIVE=True`. Cùng runtime tự tiếp tục các ID chưa hoàn tất.

Notebook 06 chỉ tải ba file cần cho training: `train_reasoning.jsonl`, `dataset.json`, `generation.json`. Dataset phải hoàn tất trên toàn train, không smoke, khớp nguyên mẫu/hash train và validation; mặc định ≥95% mẫu accepted. Không dùng val/dev/test để sinh training rationale. Mẫu rejected bị loại, không âm thầm chuyển sang answer-only. Nếu cần đổi ngưỡng sau khi kiểm tra, đặt `data.reasoning_min_coverage` trong SFT_OVERRIDES; giữ nguyên `data.reasoning_dir` trong dict override của notebook.

## Student SFT

Base **Qwen3-VL-4B-Instruct**, NF4/BF16; 2 epoch, LR peak 1e-4, eval/save mỗi 225 optimizer step như baseline. Do chuỗi dài hơn, mặc định microbatch **2 × accumulation 4 = effective batch 8**, eval batch 4. LoRA language và tùy chọn projector giữ cùng phạm vi. Prompt student chỉ có ảnh + câu hỏi, không có đáp án chuẩn; target:

```text
<think>
1. Bằng chứng quan sát được...
2. Kiến thức văn hóa liên quan...
3. Suy ra đáp án...
</think>
<answer>Đáp án chuẩn</answer>
```

Loss mặc định tính trên toàn assistant target, bao gồm reasoning và answer. Validation định kỳ vẫn dùng toàn bộ gold answer, không tạo rationale từ nhãn val; target val chỉ là `<answer>gold</answer>`. Loss này là proxy answer-only dưới teacher forcing, không đo chất lượng reasoning và không so trực tiếp với train loss có thêm reasoning.

Cuối cùng model tự sinh cả reasoning + answer bằng vLLM (mặc định tối đa 1.024 token, context 4.096); scorer chỉ nhận nội dung `<answer>`. Format thiếu/sai làm báo lỗi thay vì chấm cả thinking; output gốc nằm ở `*.traces.jsonl` để xem và được upload cùng kết quả. Có thể tăng `inference.max_new_tokens`/`inference.vllm_max_model_len` khi chạy inference nếu thực tế bị cắt. Khi đổi cấu hình generation, dùng output mới để không trộn predictions từ hai cấu hình.

Student tốt nhất: `MMCQA/task6-runs/experiment04_reasoning/sft/best_model/`. Trong notebook 04, đặt **RUN_NAME và INPUT_RUN_NAME = 'experiment04_reasoning'**. Tự đọc prompt, giới hạn ảnh và cờ reasoning từ bản merged, không cần nhập lại prompt thủ công. Student không cần tải teacher để inference.

## Kiểm chứng

CPU regression tests dùng fixture/mock để kiểm tra phân tách train/validation, retry/resume, khóa revision, hash/coverage, định dạng training, giữ prompt trong artifact, trích final answer và backend vLLM. Chưa chạy load FP8/vLLM hay training thực tế trên GPU; chưa có benchmark tốc độ, VRAM hoặc điểm cải thiện. Cần kiểm tra chất lượng lời giải trên ảnh thật trước khi đầu tư full SFT.

Nguồn triển khai: [Qwen FP8 model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-FP8), [vLLM Qwen3-VL guide](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3-VL.html), [vLLM 0.11 FP8 support](https://docs.vllm.ai/en/v0.11.0/features/quantization/fp8.html).
