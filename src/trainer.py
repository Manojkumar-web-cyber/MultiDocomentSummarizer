import numpy as np
from datasets import load_dataset
from transformers import (
    LEDForConditionalGeneration, LEDTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from evaluate import load

class LEDTrainer:
    def __init__(self, config):
        self.config = config
        self.model_name = config['model']['name']
        self.max_input_length = config['model']['max_input_length']
        self.max_target_length = config['model']['max_target_length']
        self.tokenizer = LEDTokenizer.from_pretrained(self.model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(self.model_name)
        if config['model'].get('gradient_checkpointing'):
            self.model.gradient_checkpointing_enable()
        self.rouge = load('rouge')

    def preprocess_function(self, ex):
        mi = self.tokenizer(
            ex['text'], max_length=self.max_input_length,
            padding="max_length", truncation=True)
        mi["global_attention_mask"] = []
        for ids in mi["input_ids"]:
            global_attention = [0] * len(ids)
            global_attention[0] = 1
            mi["global_attention_mask"].append(global_attention)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                ex['summary'], max_length=self.max_target_length,
                padding="max_length", truncation=True)
        mi["labels"] = labels["input_ids"]
        return mi

    def compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}

    def train(self, resume_from_checkpoint=None):
        dset = load_dataset('json', data_files={
            "train": self.config['data']['train_file'],
            "validation": self.config['data']['val_file'],
        })
        tok = dset.map(
            self.preprocess_function, batched=True,
            remove_columns=dset['train'].column_names, desc="Tokenizing"
        )
        args = self.config['training']
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['output']['model_dir'],
            num_train_epochs=args['num_epochs'],
            per_device_train_batch_size=args['per_device_train_batch_size'],
            per_device_eval_batch_size=args['per_device_eval_batch_size'],
            gradient_accumulation_steps=args['gradient_accumulation_steps'],
            learning_rate=args['learning_rate'],
            warmup_steps=args['warmup_steps'],
            weight_decay=args['weight_decay'],
            evaluation_strategy=args['evaluation_strategy'],
            eval_steps=args['eval_steps'],
            save_strategy=args['save_strategy'],
            save_steps=args['save_steps'],
            save_total_limit=args['save_total_limit'],
            load_best_model_at_end=args['load_best_model_at_end'],
            metric_for_best_model=args['metric_for_best_model'],
            greater_is_better=args['greater_is_better'],
            logging_dir=self.config['output']['logs_dir'],
            logging_steps=args['logging_steps'],
            report_to=args['report_to'],
            predict_with_generate=args['predict_with_generate'],
            generation_max_length=args['generation_max_length'],
            generation_num_beams=args['generation_num_beams'],
            fp16=args['fp16'],
            gradient_checkpointing=True,
            dataloader_num_workers=self.config['hardware']['num_workers'],
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True)
        callbacks = []
        if args['early_stopping']:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args['early_stopping_patience']))
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tok['train'],
            eval_dataset=tok['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()
