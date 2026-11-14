from ..common import config, messages, path, provenance, rows_for


def main():
    from datasets import Dataset, Image
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    from ..models.loading import load
    from .rewards import build_reward
    cfg = config()
    set_seed(cfg['training']['seed'])
    if not cfg['model'].get('adapter'):
        raise ValueError('Set model.adapter to the SFT adapter for continued GRPO training')
    # TRL 0.24 disables LoRA for its implicit KL reference. That would reference the
    # original base, NOT the SFT policy. Do not silently use that incorrect reference.
    if cfg['training'].get('beta', 0) != 0:
        raise ValueError('This continuation recipe requires beta=0. A frozen SFT reference must be implemented for KL.')
    model, processor = load(cfg, train=True)
    processor.tokenizer.padding_side = 'left'
    tc = dict(cfg['training'])
    for key in ('load_best_model_at_end', 'metric_for_best_model', 'greater_is_better'):
        tc.pop(key, None)
    tc.update(output_dir=str(path(tc['output_dir'])), remove_unused_columns=False,
              bf16=cfg['_dtype'] == 'bfloat16', fp16=cfg['_dtype'] == 'float16')
    if tc.get('max_prompt_length') is not None:
        raise ValueError('Keep max_prompt_length=null: truncating image tokens corrupts multimodal inputs')

    def dataset(split):
        records = []
        for row in rows_for(cfg, split):
            if not row.get('answer'):
                raise ValueError(f'GRPO requires labelled data: {split}')
            records.append({'prompt': messages(cfg, row['question']), 'image': str(path(row['image'])),
                            'answer': row['answer'], 'id': row['id']})
        return Dataset.from_list(records).cast_column('image', Image())

    train = dataset(cfg['data']['train_split'])
    validation = dataset(cfg['data']['eval_split']) if tc.get('eval_strategy') != 'no' else None
    args = GRPOConfig(**tc)
    reward = build_reward(cfg)  # loaded once, not for each rollout
    provenance(cfg, tc['output_dir'])
    trainer = GRPOTrainer(model=model, processing_class=processor, args=args,
                          train_dataset=train, eval_dataset=validation, reward_funcs=reward)
    trainer.train(resume_from_checkpoint=cfg.get('resume_from_checkpoint'))
    final = path(tc['output_dir']) / 'final'
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        processor.save_pretrained(final)


if __name__ == '__main__':
    main()
