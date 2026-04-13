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
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Quad_L2Net_ConfCFS', help='Model Architecture')
    args = parser.parse_args()

    # Load training datasets with a 90/10 train/val split.
    # The 10% val split is used for per-epoch evaluation with the same MMA metric
    # as the final HPatches test, but computed from optical flow correspondences.
    web_images_train = datasets.load_dataset('JimmyFu/web_images', split='train[:90%]')
    web_images_val = datasets.load_dataset('JimmyFu/web_images', split='train[90%:]')
    aachen_db_images_train = datasets.load_dataset('JimmyFu/aachen_db_images', split='train[:90%]')
    aachen_db_images_val = datasets.load_dataset('JimmyFu/aachen_db_images', split='train[90%:]')
    aachen_style_transfer_train = datasets.load_dataset('JimmyFu/aachen_style_transfer', split='train[:90%]')
    aachen_style_transfer_val = datasets.load_dataset('JimmyFu/aachen_style_transfer', split='train[90%:]')
    aachen_flow_pairs_train = datasets.load_dataset('JimmyFu/aachen_flow_pairs', split='train[:90%]')
    aachen_flow_pairs_val = datasets.load_dataset('JimmyFu/aachen_flow_pairs', split='train[90%:]')

    dataset = datasets.interleave_datasets([
        web_images_train, aachen_db_images_train,
        aachen_style_transfer_train, aachen_flow_pairs_train
    ])

    # Cap the flow eval set at 1000 samples so per-epoch evaluation is fast
    flow_eval_dataset = datasets.interleave_datasets([
        web_images_val, aachen_db_images_val,
        aachen_style_transfer_val, aachen_flow_pairs_val
    ])
    n_eval = min(1000, len(flow_eval_dataset))
    flow_eval_dataset = flow_eval_dataset.select(range(n_eval))

    # HPatches sequences — used only for the final test after training
    hpatches_eval_dataset = datasets.load_dataset('JimmyFu/hpatches_sequences', split='eval')

    image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def apply_transforms(examples):
        output_examples = {}
        for i in range(1, 7):
            output_examples[f'{i}.ppm'] = [image_transforms(img) for img in examples[f'{i}.ppm']]
            if i != 1:
                output_examples[f'h_1_{i}'] = [torch.from_numpy(np.array(h)) for h in examples[f'h_1_{i}']]
        return output_examples
    hpatches_eval_dataset.set_transform(apply_transforms)

    transform = FullTransform()
    dataset.set_transform(transform)
    flow_eval_dataset.set_transform(transform)

    # Define Model
    if args.model == 'Quad_L2Net_ConfCFS':
        model = Quad_L2Net_ConfCFS()
    elif args.model == 'ConvnextV2':
        model = ConvNeXtV2()

    # Define Sampler
    sampler = NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                          subd_neg=-8, maxpool_pos=True)

    # Define loss
    loss = MultiLoss(
        1, ReliabilityLoss(sampler, base=0.5, nq=20),
        1, CosimLoss(N=16),
        1, PeakyLoss(N=16)
    )
    output_dir = "/workspace/outputs"
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        eval_dataset=flow_eval_dataset,
        callbacks=[WeightAnalysisCallback(), EvalCallback()],
        compute_metrics=compute_metrics,
    )
    trainer.set_loss(loss.cuda())
    resume_from_checkpoint = False
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        resume_from_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Final evaluation on HPatches sequences
    print("Running final evaluation on HPatches sequences...")
    hpatches_metrics = trainer.evaluate(eval_dataset=hpatches_eval_dataset)
    print(f"HPatches MMA:         {hpatches_metrics['eval_MMA']:.4f}")
    print(f"HPatches avg matches: {hpatches_metrics['eval_avg_matches']:.1f}")
    print(f"HPatches avg feats:   {hpatches_metrics['eval_avg_feats']:.1f}")
