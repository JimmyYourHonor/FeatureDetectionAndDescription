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
import torchvision.transforms as T

if __name__ == '__main__':
    web_images = datasets.load_dataset('JimmyFu/web_images', split='train')
    aachen_db_images = datasets.load_dataset('JimmyFu/aachen_db_images', split='train')
    aachen_style_transfer = datasets.load_dataset('JimmyFu/aachen_style_transfer', split='train')
    aachen_flow_pairs = datasets.load_dataset('JimmyFu/aachen_flow_pairs', split='train')
    dataset = datasets.interleave_datasets([web_images, aachen_db_images,
                                            aachen_style_transfer, aachen_flow_pairs])
    eval_dataset = datasets.load_dataset('JimmyFu/hpatches_sequences', split='eval')
    image_transforms = T.Compose([
        T.ToTensor(),           # Convert image to PyTorch tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    def apply_transforms(examples):
        output_examples = {}
        for i in range(1, 7):
            output_examples[f'{i}.ppm'] = [image_transforms(img) for img in examples[f'{i}.ppm']]
            if i != 1:
                output_examples[f'h_1_{i}'] = [torch.from_numpy(np.array(h)) for h in examples[f'h_1_{i}']]
        return output_examples
    eval_dataset.set_transform(apply_transforms)
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
        output_dir="/content/drive/MyDrive/image_feature_model/",
        optim="adamw_torch",
        lr_scheduler_type="constant",
        learning_rate=5e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        batch_eval_metrics=True,
        num_train_epochs=25,
        weight_decay=5e-4,
        dataloader_num_workers=4,
        eval_strategy="epoch",
        save_strategy="best",
        logging_dir='./logs',
        logging_steps=100,
        remove_unused_columns=False,
        report_to="none",
        metric_for_best_model="MMA",
        # load_best_model_at_end=True,
        greater_is_better=True,
        max_grad_norm=0,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=[WeightAnalysisCallback(), EvalCallback()],
        compute_metrics=compute_metrics,
    )
    trainer.set_loss(loss.cuda())

    trainer.train()
