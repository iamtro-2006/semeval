# Actual dataset inspection

Snapshot: `e850ede73e2f1ef3dde12e0210de67d107247ef0`.

English textual QA lives only in `qa/mena/`: three JSONL plus equivalent three Parquet files. `sqa/mena/` is the spoken-question task and is excluded. `demo/` embeds media for a 24-row preview and is excluded. Other country/language tracks and all audio ZIPs are excluded.

Train/dev schema: id (string), image (relative image filename), question (English string), answer (English string). Devtest has only id, image, question. Unlike the older reference-repo loader, no country/category/subcategory columns are required or expected.

Image references use `images/<id>.jpg`. Each split has a separate shared MENA ZIP. The preparation code extracts into `data/raw/<source_split>/images/` and writes portable paths into prepared manifests. It checks ID and image disjointness across all source splits, and holds out validation by image group from train only.

See DATA_INSPECTION.json for measured counts, image dimensions, archive byte sizes and deterministic manifest hashes. No samples from devtest are permitted as SFT targets or GRPO reward references.
