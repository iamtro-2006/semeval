"""Answer-only SFT with explicit, checked multimodal prompt masking."""
from ..common import config, image, messages, path, provenance, rows_for


class QADataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class AnswerCollator:
    def __init__(self, cfg, processor):
        self.cfg, self.processor = cfg, processor

    def __call__(self, rows):
        import torch
        prompts, full, images = [], [], []
        for row in rows:
            msg = messages(self.cfg, row['question'])
            prompts.append(self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            full.append(self.processor.apply_chat_template(
                msg + [{'role': 'assistant', 'content': [{'type': 'text', 'text': row['answer']}]}],
                tokenize=False, add_generation_prompt=False))
            images.append(image(row))
        batch = self.processor(text=full, images=images, padding=True, truncation=False, return_tensors='pt')
        prefix = self.processor(text=prompts, images=images, padding=True, truncation=False, return_tensors='pt')
        labels = batch['input_ids'].clone()
        for i in range(len(rows)):
            length = int(prefix['attention_mask'][i].sum())
            if not torch.equal(batch['input_ids'][i, :length], prefix['input_ids'][i, :length]):
                raise ValueError('Chat template prompt is not an exact prefix; refusing incorrect loss masking')
            labels[i, :length] = -100
        labels[batch['attention_mask'] == 0] = -100
        if not (labels != -100).any(dim=1).all():
            raise ValueError('An example has no answer tokens')
        batch['labels'] = labels
        return batch


def main():
    from transformers import Trainer, TrainingArguments, set_seed
    from ..models.loading import load
    cfg = config()
    set_seed(cfg['training']['seed'])
    model, processor = load(cfg, train=True)
    tc = dict(cfg['training'])
    tc.update(output_dir=str(path(tc['output_dir'])), remove_unused_columns=False,
              bf16=cfg['_dtype'] == 'bfloat16', fp16=cfg['_dtype'] == 'float16')
    train = QADataset(rows_for(cfg, cfg['data']['train_split']))
    validation = QADataset(rows_for(cfg, cfg['data']['eval_split']))
    collator = AnswerCollator(cfg, processor)
    # Preflight real template, image expansion, answer masks before allocating an optimizer.
    sample = collator([train[0]])
    print('Preflight:', {k: tuple(v.shape) for k, v in sample.items()},
          'answer tokens:', int((sample['labels'] != -100).sum()))
    provenance(cfg, tc['output_dir'])
    trainer = Trainer(model=model, args=TrainingArguments(**tc), train_dataset=train,
                      eval_dataset=validation, data_collator=collator, processing_class=processor)
    trainer.train(resume_from_checkpoint=cfg.get('resume_from_checkpoint'))
    final = path(tc['output_dir']) / 'final'
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        processor.save_pretrained(final)


if __name__ == '__main__':
    main()
