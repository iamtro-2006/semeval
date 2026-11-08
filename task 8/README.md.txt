## 🚨🚨🚨 Data access 🚨🚨🚨

Access to the data requires filling in the [**Data Usage Agreement**](https://forms.gle/Pnvv96zXbnaA3Tcp8) and accepting all the terms and conditions. Once the agreement has been processed, you will be added to the task's **Google Group**, where the password to extract the archives is communicated.

The data files are distributed as a **password-protected RAR archive**. We recommend extracting it with [WinRAR](https://www.win-rar.com/) on Windows; on macOS you can use the command-line version of WinRAR or a free tool that supports encrypted RAR archives, such as [The Unarchiver](https://theunarchiver.com/) or [Keka](https://www.keka.io/); on Linux, `unrar x archive.rar`.




## Files

Training data for **StereoQueerEval** (SemEval 2027): detection of queer stereotypes and hate speech in YouTube comments, in **English, Italian and Dutch**. Persian data will be released only as part of the test set.

Each instance is a YouTube comment posted under a news video about LGBTQIA+ topics, annotated for the three subtasks. Comments are provided together with the title and description of the video they respond to, since the target of a comment is often only interpretable in context.

| File | Textual Triplets |
|---|---|
| `StereoQueerEval_EN_training.tsv` | 2,989 |
| `StereoQueerEval_IT_training.tsv` | 2,400 |
| `StereoQueerEval_NL_training.tsv` | 2,238 |

The test sets (English, Italian, Dutch _with the addition of Persian_) will be released according to the task schedule in January 2027.


## Format

UTF-8, tab-separated, one header line, fields quoted where needed.
⚠️ Titles, descriptions and comments may contain **newlines ("\n") inside quoted fields**: read the files with a proper TSV parser (e.g. `pandas.read_csv(..., sep="\t")`, Python `csv`, Hugging Face `datasets`), not with line-based tools.

## Columns

| Column | Description |
|---|---|
| `StereoQueerEval_id` | Unique instance ID, e.g. `training_EN_0001`. |
| `yt_title` | Title of the YouTube video. |
| `yt_description` | Description of the YouTube video. |
| `yt_comment` | The comment to be classified. |
| `stereotype` | Does the comment convey a stereotype about queer people? `yes` / `no`. |
| `hate_speech` | Is the comment hateful, and how? `no` / `yes_implicit` / `yes_explicit`. |
| `target` | Target of the hate: `none` if `hate_speech` is `no`; otherwise scope (`group` or `individual`) + one or more target identities, e.g. `group_lgbtqia+`, `individual_t,nb`. |

Target identities: `l`, `g`, `b`, `t`, `q`, `i`, `a`, `nb` (non-binary), `lgbtqia+` (the community as a whole), always listed in this order. `target` is present **if and only if** the comment is hateful, if the comment is not hateful then `target` must default to `none`.

## Content warning

The data contains offensive, hateful and discriminatory language towards LGBTQIA+ people, reproduced unaltered for research purposes.

## Licensing / Terms of Use

By accessing or using this dataset, you agree to the following terms.

###### 1. Permitted Use
This dataset may be used only for academic, non-commercial research, and solely for participation in the StereoQueerEval shared task or directly related research.

###### 2. No Redistribution
You may not share, publish, or redistribute the raw dataset, or any substantial portion of it, to third parties in any form. This includes, but is not limited to, public repositories, websites, and supplementary materials.

###### 3. No Commercial Use
You may not use the dataset, or any derivative of it, for commercial purposes of any kind.

###### 4. Retention
You may keep the dataset after the shared task concludes for ongoing non-commercial research, provided you continue to comply with all other terms listed here.

###### 5. Ethical Use
This dataset contains content related to LGBTQ+ communities and stereotypes. Please handle it with care and respect. You may not use the dataset, any outputs derived from it, or any models trained on it in ways that could target, harm, discriminate against, or demean LGBTQ+ individuals or communities.

## Baseline inference

Model output uses only `target` and `reasoning` (one short English sentence).
Run `python src/inference.py` from this directory, or open
`notebooks/inference.ipynb` in a Jupyter/IPython environment. The notebook calls
the same pipeline; choose the mode, input fields and split, then enable
`RUN_INFERENCE` to run the model. Supply the private TSV files configured in
`src/configs/inference.json`.

New prediction files store `reasoning` instead of `evidence_json`; target metrics
remain independent of schema diagnostics. See `PLAN.md` for the protocol.
Run regression checks with `python -m unittest discover -s tests -v`.

## OpenJev fine-tuning on Colab

Open `openjev/train.ipynb`, or use:
[Open in Colab](https://colab.research.google.com/github/iamtro-2006/semeval/blob/main/task%208/openjev/train.ipynb).

The notebook clones `main` into `/content/semeval`, installs the training libraries,
and accepts TRAIN, DEV, optional TEST, and OUTPUT paths. Upload TSV files directly
to the session; no Drive mount or local GPU is needed. On Kaggle the clone uses
`/kaggle/working/semeval`; enable Internet and set dataset paths accordingly.

The `openjev/` folder contains `train.ipynb`, `inference.ipynb`, `run.py`,
`questions.json` and `samples.json`. Fine-tuning uses BF16 LoRA on OpenJev Flash 9B, without quantizing
the base weights. A GPU forward/backward and adapter save/reload check run in the
notebook. Actual 9B GPU execution has not been validated on the local CPU machine.

Synthetic examples are original fictional assistant-authored examples, not
official annotations: four a-positive examples and two negative controls per
train/dev split. They are added only in memory. Original dev determines checkpoint
selection and one global identity threshold; synthetic dev is reported separately.
Probabilities are restricted A/B softmax scores, without domain calibration.
Empty identity selections are abstentions, counted as invalid targets rather than
silently assigned none. Existing original split boundaries, including shared videos,
are preserved and their overlap is reported.

OUTPUT contains only `adapter/`, one `checkpoint/`, `settings.json`, `history.csv`,
`predictions.csv`, and `metrics.json`. The final notebook cell exports these as a ZIP.
Download it before the session ends. To resume, restore its contents into OUTPUT,
keep the same settings/data/library versions/code and set RESUME=True. Test can be
added later for final evaluation; it never selects the checkpoint or threshold.

## OpenJev zero-shot and few-shot inference

Open `openjev/inference.ipynb`, or use:
[Inference in Colab](https://colab.research.google.com/github/iamtro-2006/semeval/blob/main/task%208/openjev/inference.ipynb).

Upload the original local splits from `data/split/en`, keeping the filenames:

| File | Rows | Role |
|---|---:|---|
| StereoQueerEval_EN_train.tsv | 1,107 | Few-shot example source; unnecessary for zero-shot |
| StereoQueerEval_EN_val.tsv | 316 | DATA for development experiments |
| StereoQueerEval_EN_test.tsv | 159 | DATA for final held-out evaluation |

Choose MODE="zero_shot" or MODE="few_shot". Both use the published BF16 checkpoint,
without task fine-tuning, adapters or boosting. DATA defaults to the original val
file; few-shot additionally reads the original train file.

EXAMPLES="balanced" is the few-shot default: four demonstrations per question,
two Yes/two No, and two individual/two group for scope. It prioritizes eligible
examples from the old Qwen prompt, then original train additions. The inspected
bisexual and transgender additions are preferred over other additions. Other
train candidates favor shorter inputs. Original gold labels are preserved.

Balanced mode excludes all ID, normalized-comment and video overlap with DATA/DEV,
including the old video's exception. If fewer than two positive/negative original
train examples remain, it uses explicitly marked synthetic train examples from
samples.json. Existing a train samples are reusable. The new few_shot section
contains q/intersex/non-binary fallback examples only; these are never added to
fine-tuning or dev/test. No synthetic dev records are used as demonstrations.
If balance cannot be reached without evaluation overlap, the job fails instead
of inventing or relabeling records. This is an experimental variant, not a claim
that artificial examples establish real-world quality.

EXAMPLES="original" retains all six examples from
src/prompts/comment_title_desc/fs_prompt.md, their original gold and order.
It uses the Qwen provenance/text/gold validator and preserves the declared
training_EN_0991 video-overlap exception, never ID/comment overlap. The old
examples have no positive q/i/a/nb demonstrations. The umbrella label does not
count as a-positive.

Each job prints source IDs, answer counts, and synthetic provenance per question.
settings.json stores the examples policy, selected IDs/targets, old seed pool,
train/prompt/sample hashes and origins. Synthetic use is also declared in the
metrics protocol. The same demonstrations are used for dev and DATA in one run;
selection depends on which evaluation videos must be excluded.

The default identity threshold is 0.5. DATA may omit targets: predictions are saved
but metrics are marked not_evaluated. Optionally enable TUNE_ON_DEV with a separate
labeled DEV file: use DATA=test and DEV=val for final evaluation. Never select
thresholds or examples using test labels. Compare configurations on the same
original dev first and keep the test held out from prompt selection.

The notebook exports only settings.json, predictions.csv and metrics.json, with
ZIP download. Few-shot OUTPUT includes balanced/original in its name so runs stay
separate; zero-shot keeps zero-shot-v2. Choose a new output folder for repeats.
No dataset copies, adapters, optimizer states or training checkpoints are created
by inference.

Prompt v2 definitions are unchanged by example balancing. The four-example
balanced variant and the historical six-example original variant use the same
question wording, A/B readout and decoding. Only the demonstrations differ.
Zero-shot has no demonstrations. Training augmentation still uses the existing
four a-positive/two negative train records, with synthetic dev reported separately.
All behavior checks run on CPU; actual OpenJev quality requires GPU evaluation.
