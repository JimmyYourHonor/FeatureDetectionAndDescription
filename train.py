from preprocessing import *
from models import *
from callbacks.weight_analysis_callback import WeightAnalysisCallback
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import datasets
from transformers import TrainingArguments

if __name__ == '__main__':
    web_images = datasets.load_dataset('JimmyFu/web_images', split='train')
    aachen_db_images = datasets.load_dataset('JimmyFu/aachen_db_images', split='train')
    aachen_style_transfer = datasets.load_dataset('JimmyFu/aachen_style_transfer', split='train')
    aachen_flow_pairs = datasets.load_dataset('JimmyFu/aachen_flow_pairs', split='train')
    dataset = datasets.interleave_datasets([web_images, aachen_db_images,
                                            aachen_style_transfer, aachen_flow_pairs])
    transform = FullTransform()
    dataset.set_transform(transform)

    # Define Model
    model = Quad_L2Net_ConfCFS()

    # Define Sampler
    sampler = NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                          subd_neg=-8,maxpool_pos=True)

    # Define loss
    loss = MultiLoss(
        1, ReliabilityLoss(sampler, base=0.5, nq=20),
        1, CosimLoss(N=16),
        1, PeakyLoss(N=16)
    )
    
    training_args = TrainingArguments(
        output_dir="./image_feature_model",
        optim="adamw_torch",
        lr_scheduler_type="constant",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=25,
        weight_decay=5e-4,
        dataloader_num_workers=4,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
        remove_unused_columns=False,
        report_to="none",
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[WeightAnalysisCallback()],
    )

    trainer.train()
