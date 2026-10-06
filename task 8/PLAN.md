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

Dùng trực tiếp **train/dev/test TSV**. Theo yêu cầu bổ sung của bạn, đã
tạm chia raw EN vào `data/split/en`: train 2.391, dev 299, test 299 (80/10/10).
Đây là local split tạm thời, không phải official dev/test; có thể thay bằng
các split chính thức sau này. Inference không tự chia lại dữ liệu. Config
chứa đường dẫn của cả ba file; `data.split` quyết định file được chạy.

Split tạm giữ nguyên nhóm title+description và normalized comment trùng,
đồng thời tối ưu cân bằng scope, identity và target labels. Hai identity
`i` (14 mẫu trong một nhóm video) và `a` (1 mẫu) giữ trong train; dev/test
không thể đánh giá chúng độc lập mà vẫn giữ ranh giới video. Distribution
table/charts nằm trong `outputs/en_split`; manifests ghi phương pháp,
seed=42, source hash và kiểm tra không giao ID/video proxy/comment.

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
    split/en/                     # split EN tạm; có thể thay file chính thức
      train.tsv
      dev.tsv
      test.tsv
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

| input_variant | Trường model được đọc và trích bằng chứng |
|---|---|
| comment | yt_comment |
| comment_title | yt_comment, yt_title |
| comment_title_desc | yt_comment, yt_title, yt_description |

Input JSON không chứa ID, gold `target`, `hate_speech` hoặc `stereotype`.
Parser từ chối evidence lấy từ trường không được đưa vào baseline đó.

Output thống nhất theo thứ tự:

1. `entities`: người/cộng đồng bị nhắm tới, reference evidence và targeting
   evidence từ comment. Reference có thể cần title/description để giải
   pronoun hoặc diễn đạt gián tiếp.
2. `scope`: nhãn group/individual cùng các cặp entity–evidence chứng minh
   phạm vi; không suy scope chỉ từ title của video.
3. `identities`: đủ chín code. Mỗi code có các cặp entity–evidence hoặc
   chuỗi `none` nếu không có cặp được hỗ trợ.
4. `target`: ghép scope với đúng các identity đã có bằng chứng.

Ví dụ khái niệm: `l` phải gắn với một người/nhóm cụ thể, kèm đoạn text
chứng minh lesbian; không chỉ trả `l` vì video có nhắc từ đó. `g`, `b`,
`t`, `q`, `i`, `a`, `nb`, umbrella dùng cùng nguyên tắc.

Nếu không có hateful target hoặc không đủ evidence để xác định scope và
identity, trả `target=none`; entities rỗng, scope và mọi identity là none.
Đây bao gồm yêu cầu abstain khi thiếu bằng chứng của bạn. Với dữ liệu gold,
một trường hợp hate nhưng model không tìm được evidence có thể bị chấm sai
vì model trả none; vẫn giữ prediction đó để đánh giá trung thực.

Bằng chứng là quote nguyên văn, liên tục trong field gốc, giữ ngôn ngữ
đầu vào. Parser kiểm tra quote tồn tại và entity/code/target nhất quán;
việc quote đúng có thực sự chứng minh identity hay không vẫn cần đọc đánh
giá về nghĩa, không được xem kiểm tra substring là bảo đảm semantic accuracy.

Few-shot hiện dùng cùng sáu mẫu từ **train EN hiện tại** cho cả ba baseline:

| Source ID | Gold target | Vai trò minh họa |
|---|---|---|
| training_EN_1057 | none | Ủng hộ cộng đồng, không suy hate từ identity mention |
| training_EN_1226 | individual_l | Một người; chỉ baseline đầy đủ mới gọi tên Barbara Johnson từ description |
| training_EN_0396 | group_t | Identity denial đối với trans women |
| training_EN_0003 | group_l,g | Comment khái quát về homosexual people |
| training_EN_1776 | group_lgbtqia+ | Target là cộng đồng umbrella |
| training_EN_0428 | group_t,nb | Multi-label theo annotation hiện có của train |

Text và target lấy nguyên từ train; các cặp entity–evidence được biên soạn
thủ công vì TSV không có gold evidence. Giữ nguyên gold giữa ba baseline;
chỉ thay text được cung cấp và evidence có thể truy cập. Annotation có thể
cần review về nghĩa, đặc biệt với identity denial hoặc multi-label.

Trước khi load model, code kiểm tra source ID thuộc train đã cấu hình,
text khớp các trường train, target khớp gold, quote tồn tại, và demonstration
không trùng ID/video proxy/comment với tập đang chấm. Few-shot nên chạy
dev/test; chạy trên cả train hiện tại sẽ bị từ chối vì chứa chính các demo.
Khi thay train hoặc chuyển IT/NL, cập nhật block `examples` bằng các mẫu
từ train tương ứng; chỉ đổi `<language>` không tự dịch hoặc chọn lại demo.
Không cần sửa pipeline. `tools/build_baseline_prompts.py` là tiện ích soạn
lại sáu file cho train EN này, không tự chạy trong inference.

## 7. Inference và outputs

Các bước trên server:

1. Đọc config và TSV đã được chọn, bằng parser hỗ trợ quoted newline.
2. Chọn prompt theo baseline/mode, chỉ đưa các trường được phép vào input;
   kiểm tra nguồn train của few-shot trước khi thay language và dựng messages.
3. Load tokenizer/model qua hai Transformers Auto classes.
4. Dựng chat messages bằng tokenizer chat template. Few-shot có các cặp
   user/assistant ví dụ trước query cuối.
5. Generate evidence annotations và target cùng một response JSON.
6. Validate schema, quote trong trường hiển thị và quan hệ entity–label;
   ghi prediction aligned ID.
7. Nếu có gold target, chấm metrics; nếu không có gold, ghi not_evaluated.

Input budget mặc định 8.192 tokens và output budget 1.536 tokens vì output
có bằng chứng, không chỉ một nhãn. Bạn chỉnh trong config. Không tự cắt
text làm mất bằng chứng: mẫu vượt budget được ghi lỗi và giữ trong output.
JSON không hợp lệ/quote không tồn tại cũng được giữ nguyên raw response,
prediction rỗng và trạng thái invalid; không tự sửa thành none hoặc bỏ dòng.

`predictions.tsv` lưu ID, language, input_variant, prompt_mode, gold nếu có, pred_target, valid/error,
evidence_json, raw_output, input/output tokens và latency.
`run.json` lưu config, hashes dữ liệu/prompt/train, demo IDs, trường input
được dùng và model revision. Mỗi baseline/mode/split nằm trong thư mục riêng.

`metrics.json` có exact match trên toàn bộ dữ liệu kể cả none, invalid-rate,
false-positive target trên gold none, và bộ `src/metrics.py` trên gold có
target. Bộ metrics sẵn có không chấm gold none nên phần này được gọi rõ là
conditional Task C report, không gọi là toàn bộ full-target score. Không có
gold evidence trong TSV hiện tại nên chưa tính F1 cho evidence extraction.

## 8. Trình tự nghiên cứu

| Bước | Việc cần thực hiện |
|---|---|
| 1 | Bạn thêm train/dev/test; chỉnh paths, language, repo ID và GPU settings |
| 2 | Review prompt và các cặp evidence; chọn few-shot examples từ train |
| 3 | Cài một môi trường requirements trên server và chạy inference trên dev |
| 4 | So sánh zero/few-shot với 3B, 7B, 14B, 32B; dùng cùng data và protocol |
| 5 | Nếu nghiên cứu GPTQ, thay repo ID và so với bản thường cùng size, budget và prompt |
| 6 | Phân tích lỗi none, group/individual, từng identity, multi-label và evidence |
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
demo, quotes, và việc tách demo khỏi dev/test bằng dữ liệu hiện có; chưa
load model, đo token thực tế hoặc chạy inference trên GPU.
Inference không tự chia lại dữ liệu. Giữ dữ liệu và output có trích văn bản
ở môi trường nghiên cứu riêng theo điều khoản README.
