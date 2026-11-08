# Kế hoạch triển khai Qwen2.5 cho nhãn `target`

## 1. Bài toán và dữ liệu

Đã đọc `README.md.txt`, `ACRONYMS.md.txt`, `src/metrics.py` và các mẫu raw
EN. README xác nhận đây là StereoQueerEval, SemEval 2027, với train bằng
English, Italian và Dutch. Tên hai tài liệu trong workspace có đuôi `.md.txt`.

Mỗi mẫu có `yt_title`, `yt_description`, `yt_comment`. Comment là đối tượng
phân loại; title/description cung cấp context để xác định người hoặc cộng
đồng được nhắc đến. Ba cột annotation là `stereotype`, `hate_speech`,
`target`; pipeline này chỉ dự đoán `target`.

Nhãn `target`:

- `none` nếu comment không có hate, theo README.
- Nếu có target: `individual` hoặc `group`, nối với một hoặc nhiều mã
  identity: `l,g,b,t,q,i,a,nb,lgbtqia+`, theo đúng thứ tự này.
- Ví dụ: `individual_l`, `group_t`, `group_l,g`, `group_t,nb`.
- Đây là scope classification kết hợp multi-label identity prediction,
  không phải chọn một identity duy nhất.

ACRONYMS giải thích `l/g/b/t/q/i/a`; README bổ sung `nb` và code umbrella
`lgbtqia+`. `a` dùng cho asexual/aromantic/agender; không tự suy ra allies.

| Raw training | Số mẫu | `none` | Có target |
|---|---:|---:|---:|
| EN | 2.989 | 1.407 | 1.582 |
| IT | 2.400 | 1.160 | 1.240 |
| NL | 2.238 | 1.127 | 1.111 |

EN có 356 target individual, 1.226 target group. Support của
`l/g/b/t/q/i/a/nb/lgbtqia+` lần lượt là
133/333/59/772/8/14/1/15/465. Các nhãn hiếm phải được đọc cùng support;
không kết luận chất lượng từ một điểm Macro-F1 đơn lẻ.

Ví dụ EN: `training_EN_0001` dùng “She's…” nên cần context để xác định
Barbara Johnson và identity lesbian, gold `individual_l`.
`training_EN_0003` là video về một người gay nhưng comment khái quát về
homosexuality, gold `group_l,g`. `training_EN_0007` nhắc gay trong phát
biểu ủng hộ equality, gold `none`. Vì vậy nhắc identity không đủ để tạo target.

## 2. Phạm vi triển khai

Dùng trực tiếp **train/val/test TSV mới (v4)** trong `data/split/en`:
train 1.107, val 316, test 159 (gần 70/20/10). Chỉ giữ các dòng
`hate_speech=yes_explicit` hoặc `yes_implicit`; cả ba split không có
gold `target=none`. Train có 861 group và 246 individual.

Đây là split local từ raw training, không phải official dev/test. V4 chia
theo toàn bộ chuỗi target với seed=42; nhãn chỉ có một mẫu được đưa vào
test. V4 không giữ nguyên nhóm video, nên video có thể xuất hiện ở nhiều
split. Mô tả dữ liệu hiện tại nằm trong `data/split/data_overview_v4.md`;
chart cũ trong `outputs/en_split` mô tả split trước, không phải v4.

Inference không tự chia lại dữ liệu. Config chứa đường dẫn ba file mới;
`data.split=dev` ánh xạ tới `StereoQueerEval_EN_val.tsv`. Các demo được
chọn từ train sao cho không giao ID/video proxy/comment với val/test.

Không cài packages, load model hoặc chạy smoke test trên máy hiện tại.
Code và tài liệu được chuẩn bị để bạn cài và chạy trên server. Chưa có
prediction hoặc score của Qwen; các output template chỉ mô tả cấu trúc.

Một file `requirements.txt`, một config inference, ba thư mục prompt.
Mỗi thư mục có `zs_prompt.md` và `fs_prompt.md` với instruction tiếng Anh.
Không có registry model, config riêng theo kích thước hoặc nhánh pipeline
riêng cho GPTQ.

## 3. Cấu trúc

```text
task 8/
  PLAN.md
  requirements.txt
  README.md.txt
  ACRONYMS.md.txt
  data/
    raw/                          # tài liệu dữ liệu gốc
    split/en/                     # v4: chỉ các mẫu hate speech
      StereoQueerEval_EN_train.tsv
      StereoQueerEval_EN_val.tsv   # data.split=dev
      StereoQueerEval_EN_test.tsv
  src/
    configs/inference.json
    prompts/
      comment/
        zs_prompt.md
        fs_prompt.md
      comment_title/
        zs_prompt.md
        fs_prompt.md
      comment_title_desc/
        zs_prompt.md
        fs_prompt.md
    inference.py
    metrics.py                    # giữ bộ chấm sẵn có
  outputs/
    predictions.template.tsv
    metrics.template.json
    <model>/<input_variant>/<prompt_mode>/<split>/<run_id>/
      predictions.tsv
      metrics.json
      run.json
```

## 4. Cấu hình và load model

`src/configs/inference.json` có một trường `model_id`. Dán trực tiếp repo
ID vào trường đó, ví dụ `Qwen/Qwen2.5-7B-Instruct` hoặc
`Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`. Cùng cách này dùng cho 3B/14B/32B.
Pipeline chỉ sử dụng:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id, ...)
model = AutoModelForCausalLM.from_pretrained(model_id, ...)
```

Transformers đọc thông tin checkpoint; không tự xây registry, không tự
quantize weights và không chọn backend bằng heuristic theo tên repo.
Đổi repo ID không thay đổi các bước đọc dữ liệu, dựng prompt, generate,
parse hay ghi output.

Các thiết lập bạn chỉnh trong config:

| Trường | Ý nghĩa |
|---|---|
| model_id | Repo ID hoặc đường dẫn checkpoint local |
| input_variant | comment, comment_title hoặc comment_title_desc |
| model.dtype | auto, bfloat16, float16… tùy server |
| model.device_map | auto hoặc mapping thiết bị bạn muốn |
| model.max_memory | null hoặc ví dụ `{"0":"20GiB","1":"20GiB","cpu":"64GiB"}` |
| model.revision | Revision model; có thể cố định commit để tái lập |
| model.attention_implementation | sdpa mặc định |
| model.local_files_only | Dùng model đã tải sẵn khi true |
| data.language | English / Italian / Dutch; thay `<language>` trong prompt |
| data.split | train, dev hoặc test |
| data.files | Đường dẫn ba TSV đã có sẵn |
| prompt.mode | zero_shot hoặc few_shot |
| prompt.directory | Thư mục chứa ba baseline; code tự chọn file theo input_variant và mode |
| prompt.shots | Số ví dụ few-shot lấy theo thứ tự trong file; hiện hỗ trợ 1–6, mặc định 6 |
| generation | seed, input/output token budget, do_sample |
| output_dir | Thư mục lưu runs |

Đường dẫn tương đối tính từ thư mục `task 8`. Chọn GPU/VRAM trong config
và môi trường server; kích thước model lớn vẫn cần đủ bộ nhớ. `max_memory`
giới hạn bộ nhớ phân bố model, không làm giảm kích thước checkpoint.

## 5. Một file requirements

`requirements.txt` gồm PyTorch, Transformers, Accelerate, Safetensors,
Hub/Jinja2, cùng runtime GPTQModel/Optimum/Ninja để hỗ trợ checkpoint GPTQ
qua chính `AutoModelForCausalLM`. TSV/JSON và metrics dùng Python stdlib.

Đây là danh sách dependency đề xuất cho server, chưa được cài hoặc xác
minh trên máy này. Các version core/GPTQ được chọn theo dependencies
upstream; cần dùng CUDA wheel phù hợp với driver và GPU của server.

Việc model card có ví dụ load ngắn không có nghĩa checkpoint lượng tử hóa
không cần runtime bổ sung. [Transformers hướng dẫn GPTQModel cho GPTQ](https://huggingface.co/docs/transformers/v5.17.0/en/quantization/gptq).
Runtime được cài trong cùng file requirements; code load vẫn như trên.

Cài trên server, từ root project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CUDA 12.8 trong ví dụ cần được bạn thay nếu server dùng profile khác;
[PyTorch có các CUDA wheel tương ứng](https://pytorch.org/get-started/previous-versions/).
Driver/CUDA Toolkit/compiler là thành phần hệ thống khi backend cần,
không được cài bằng requirements. Không yêu cầu chạy các lệnh này cục bộ.

## 6. Thiết kế prompt: thực thể và bằng chứng trước nhãn

Mỗi baseline có hai file `zs_prompt.md` và `fs_prompt.md`; instruction đều
bằng tiếng Anh, có `<language>` và `<input_json>`. Dùng chung train/dev/test
TSV cho cả ba baseline, không tạo dữ liệu con. Code lọc các trường trước
khi dựng messages, cho cả query và ví dụ few-shot:

| input_variant | Trường model được đọc |
|---|---|
| comment | yt_comment |
| comment_title | yt_comment, yt_title |
| comment_title_desc | yt_comment, yt_title, yt_description |

Input JSON không chứa ID, gold `target`, `hate_speech` hoặc `stereotype`.
Output chỉ gồm hai trường:

1. `target`: `none` hoặc `<scope>_<identity_codes>`, ví dụ `individual_t`,
   `group_l,g`. Giữ quy ước scope và thứ tự identity hiện có.
2. `reasoning`: một câu tiếng Anh ngắn giải thích đối tượng và lý do chọn nhãn;
   nêu liên kết ngữ cảnh khi cần. Không yêu cầu trích dẫn nguyên văn.

Không xuất object scope, danh sách identity hoặc evidence riêng. Nếu không có
hateful target hoặc thiếu bằng chứng để xác định scope và identity, trả
`{"target":"none","reasoning":"No supported hateful target can be identified."}`.
Validator kiểm tra đúng hai key, target hợp lệ theo thứ tự chuẩn và reasoning
là chuỗi không rỗng. Tính đúng về nghĩa cần đánh giá riêng.

Few-shot giữ nguyên sáu mẫu trong mỗi file prompt hiện tại. Source ID, nhãn,
reasoning và ngoại lệ video overlap được lưu trong
`tools/baseline_demonstrations.json`; input được lấy lại từ train khi chạy
`tools/build_baseline_prompts.py`. Tiện ích kiểm tra cả ba cấu hình trước khi
ghi đè sáu prompt. Không tự chạy tiện ích này trong inference.

Trước khi load model, pipeline kiểm tra nguồn train, text, gold target,
schema và overlap ID/comment/video với tập đang chấm. Giữ nguyên ngoại lệ
video overlap đã được khai báo cho tối đa một demo có số identity lớn nhất;
không cho phép trùng ID hoặc comment. Few-shot chạy trên train sẽ bị từ chối
nếu tập đó chứa demo. Train hiện tại không có none; không lấy ví dụ từ dev/test
để bổ sung. Chuyển ngôn ngữ cần chọn lại demo từ train tương ứng.

## 7. Inference và outputs

Các bước trên server:

1. Đọc config và TSV, hỗ trợ quoted newline.
2. Chọn prompt và lọc các trường input; kiểm tra nguồn few-shot.
3. Load tokenizer/model bằng Transformers và dựng chat messages.
4. Generate JSON với target và reasoning.
5. Đọc target để chấm điểm, kiểm tra schema độc lập để ghi diagnostics.
6. Lưu predictions, metrics và manifest; giữ mọi dòng kể cả output lỗi.

Input budget mặc định 8.192 tokens; output budget giảm xuống 256 tokens cho
nhãn và một câu lý do. Không tự cắt input: mẫu vượt budget được ghi lỗi.
JSON lỗi vẫn được giữ trong raw_output. Cơ chế cứu nhãn hiện có chỉ đọc một
trường target dạng JSON string rõ ràng; không suy từ gold hoặc đổi lỗi thành none.
Một nhãn hợp lệ vẫn được chấm nếu schema hoặc reasoning sai; annotation_valid
phản ánh lỗi định dạng độc lập.

`predictions.tsv` lưu ID, language, input_variant, prompt_mode, gold, pred_target,
valid/error, annotation_valid/annotation_error, reasoning, raw_output,
input/output tokens và latency. Cột reasoning thay cho evidence_json ở các run
mới; output cũ giữ nguyên. `run.json` thêm output_schema, cùng config, hashes,
demo IDs, input fields và model revision. Mỗi run có thư mục riêng.

`metrics.json` chỉ chấm target: exact match toàn bộ, invalid rate, false positive
trên gold none và Task C trên gold-active. Không chấm chất lượng reasoning.
Split v4 chỉ có gold-active, nên không đánh giá khả năng phân biệt hate/non-hate.

Notebook `notebooks/inference.ipynb` cho phép chọn model, input variant, mode,
split và shots; kiểm tra dữ liệu trước khi load model, rồi gọi cùng pipeline.
Config riêng lưu trong outputs/notebook; notebook hiển thị metrics và số lỗi
schema. Đặt RUN_INFERENCE = True sau khi kiểm tra config. Notebook không kèm
dữ liệu hoặc output thực. Môi trường notebook cần Jupyter/IPython, ngoài
requirements inference; không cài hoặc tải model tự động khi mở notebook.

## 8. Trình tự nghiên cứu

| Bước | Việc cần thực hiện |
|---|---|
| 1 | Bạn thêm train/dev/test; chỉnh paths, language, repo ID và GPU settings |
| 2 | Review prompt và lý do chọn nhãn; chọn few-shot examples từ train |
| 3 | Cài một môi trường requirements trên server và chạy inference trên dev |
| 4 | So sánh zero/few-shot với 3B, 7B, 14B, 32B; dùng cùng data và protocol |
| 5 | Nếu nghiên cứu GPTQ, thay repo ID và so với bản thường cùng size, budget và prompt |
| 6 | Phân tích lỗi none, group/individual, từng identity, multi-label và lý do chọn nhãn |
| 7 | Cố định prompt/config/model rồi chạy test; không tune bằng test |

Matrix đầy đủ EN: 4 kích thước × 2 precision × 3 input baselines × 2 prompt modes = 48 runs,
nếu bạn muốn đánh giá cả bản thường và GPTQ. Mở IT/NL bằng config và
demonstrations tương ứng. Không cần chạy toàn bộ matrix trước khi bạn chốt.

Chạy từ thư mục `task 8` trên server. Lệnh mặc định dùng cấu hình trong file:

```bash
python src/inference.py --config src/configs/inference.json
```

Có thể chọn baseline/mode/split trực tiếp mà không sửa config:

```bash
python src/inference.py --input-variant comment --prompt-mode zero_shot --split dev
python src/inference.py --input-variant comment_title --prompt-mode zero_shot --split dev
python src/inference.py --input-variant comment_title_desc --prompt-mode zero_shot --split dev
python src/inference.py --input-variant comment --prompt-mode few_shot --split dev
python src/inference.py --input-variant comment_title --prompt-mode few_shot --split dev
python src/inference.py --input-variant comment_title_desc --prompt-mode few_shot --split dev
```

Mỗi lần chạy tự ghi predictions và metrics vào thư mục output riêng.
Đã kiểm tra cú pháp Python, đúng field filtering, nguồn/text/gold của sáu
demo, schema hai trường và overlap với dev/test bằng dữ liệu hiện có; chưa
load model, đo token thực tế hoặc chạy inference trên GPU.
Inference không tự chia lại dữ liệu. Giữ dữ liệu và output có trích văn bản
ở môi trường nghiên cứu riêng theo điều khoản README.
