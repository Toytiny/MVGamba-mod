import tyro
import time
import random
import datetime
import torch
from core.options import AllConfigs
# >>> 用新的模型类（你也可以在 __init__.py 里做 alias）
from core.mvgamba_models2 import MVGamba2 as MVGamba

from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file
from core.utils import CosineWarmupScheduler
import psutil

import os
from torch.utils.tensorboard import SummaryWriter
import wandb


def needs_decay(param_name):
    if "pos_embed" in param_name:
        return False
    return True


def main():
    set_seed(42)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    opt = tyro.cli(AllConfigs)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    # -----------------------
    # model
    # -----------------------
    model = MVGamba(opt)

    # -----------------------
    # data
    # -----------------------
    if opt.data_mode == 's3':
        from core.provider_ikun2 import ObjaverseDataset2 as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # -----------------------
    # optimizer / scheduler
    # -----------------------
    parameters = [{
        "params": [param for name, param in model.named_parameters() if needs_decay(name)],
        "weight_decay": opt.weight_decay,
    }, {
        "params": [param for name, param in model.named_parameters() if not needs_decay(name)],
        "weight_decay": 0.0,
    }]

    optimizer = torch.optim.AdamW(parameters, lr=opt.lr, betas=(0.9, 0.95))
    if accelerator.is_main_process:
        print(f"model parameters: {sum(p.numel() for p in model.parameters())}")

    steps_per_epoch = max(1, len(train_dataloader) // opt.gradient_accumulation_steps)
    total_steps = opt.num_epochs * steps_per_epoch
    warmup_iters = opt.warmup_epochs * steps_per_epoch
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_iters=warmup_iters,
        max_iters=total_steps,
        min_lr=0.1 * opt.lr
    )

    # -----------------------
    # resume
    # -----------------------
    resume_after_prepare = False
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
            model.load_state_dict(ckpt['model'])
            start_epoch = ckpt.get('epoch', -1) + 1
        elif opt.resume.endswith('.pth'):
            ckpt = torch.load(opt.resume, map_location='cpu')
            print(f"resume from {opt.resume}, loading ...")
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # 兼容两种 scheduler 保存方式
            if 'scheduler' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    scheduler.scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', -1) + 1
            print("load checkpoint done!")
        else:
            resume_after_prepare = True
            start_epoch = 0
    else:
        start_epoch = 0

    # -----------------------
    # accelerate wrap
    # -----------------------
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    if resume_after_prepare:
        # 如果用 accelerate 的 save_state 恢复
        accelerator.load_state(opt.resume, strict=False)
        # 估算起始 epoch
        start_epoch = int(getattr(scheduler, "scheduler", scheduler)._step_count // steps_per_epoch) + 1
        print(f"Resuming from {opt.resume} at epoch {start_epoch}")

    # 旧代码里有 gumbel_softmax 的温度更新，这里模型不再需要（保留保护）
    if getattr(opt, "use_gumbel_softmax", False) and hasattr(model, "module"):
        num_updates = max(0, start_epoch * len(train_dataloader))
        print(f"set gumbel_softmax num_updates: {num_updates}")
        if hasattr(model.module.model.decoder, "get_rot"):
            model.module.model.decoder.get_rot.set_num_updates(num_updates)

    start_time = datetime.datetime.now()

    # -----------------------
    # logging (TB & wandb)
    # -----------------------
    if accelerator.is_main_process:
        writer = SummaryWriter(opt.workspace)
        wandb.init(
            project="radar_training",
            config=vars(opt) if hasattr(opt, "__dict__") else {},
            dir=opt.workspace,
            name=opt.workspace.split('/')[-1],
        )
        wandb.watch(accelerator.unwrap_model(model), log_freq=1000)

    # -----------------------
    # loop
    # -----------------------
    for epoch in range(start_epoch, opt.num_epochs):
        model.train()
        total_loss = 0.0
        total_psnr = 0.0  # 兼容字段，模型会返回 0
        total_loss_cd = 0.0
        total_loss_rcs = 0.0
        total_loss_vrel = 0.0

        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                step_ratio = (epoch + i / max(1, len(train_dataloader))) / max(1, opt.num_epochs)

                out = model(data, epoch, step_ratio)

                loss = out['loss']
                psnr = out.get('psnr', torch.tensor(0.0, device=loss.device))

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                # metrics 聚合
                total_loss += loss.detach()
                total_psnr += psnr.detach()
                if 'loss_cd' in out:
                    total_loss_cd += out['loss_cd'].detach()
                if 'loss_rcs' in out:
                    total_loss_rcs += out['loss_rcs'].detach()
                if 'loss_vrel' in out:
                    total_loss_vrel += out['loss_vrel'].detach()

            # 控制台日志（不再保存图像）
            if accelerator.is_main_process and (i % 100 == 0):
                mem_free, mem_total = torch.cuda.mem_get_info()
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                elapsed = datetime.datetime.now() - start_time
                elapsed_str = str(elapsed).split('.')[0]
                process = psutil.Process()
                print(
                    f"[{current_time} INFO] {i}/{len(train_dataloader)} | "
                    f"Elapsed: {elapsed_str} | "
                    f"Mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G | "
                    f"LR: {scheduler.get_last_lr()[0]:.7f} | "
                    f"Step ratio: {step_ratio:.4f} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Memory per process: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB"
                )

        # --- gather & log ---
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_loss_cd = accelerator.gather_for_metrics(total_loss_cd).mean()
        total_loss_rcs = accelerator.gather_for_metrics(total_loss_rcs).mean()
        total_loss_vrel = accelerator.gather_for_metrics(total_loss_vrel).mean()

        if accelerator.is_main_process:
            num_batches = max(1, len(train_dataloader))
            total_loss /= num_batches
            total_psnr /= num_batches
            total_loss_cd /= num_batches
            total_loss_rcs /= num_batches
            total_loss_vrel /= num_batches

            accelerator.print(
                f"[train] epoch: {epoch} "
                f"loss: {total_loss.item():.6f} "
                f"cd: {total_loss_cd.item():.6f} "
                f"rcs: {total_loss_rcs.item():.6f} "
                f"vrel: {total_loss_vrel.item():.6f} "
                f"psnr(dbg): {total_psnr.item():.4f}"
            )
            writer.add_scalar("Loss/train_total", total_loss, epoch)
            writer.add_scalar("Loss/train_cd", total_loss_cd, epoch)
            writer.add_scalar("Loss/train_rcs", total_loss_rcs, epoch)
            writer.add_scalar("Loss/train_vrel", total_loss_vrel, epoch)
            writer.add_scalar("DBG/psnr", total_psnr, epoch)

            wandb.log({
                "Loss/train_total": total_loss.item(),
                "Loss/train_cd": total_loss_cd.item(),
                "Loss/train_rcs": total_loss_rcs.item(),
                "Loss/train_vrel": total_loss_vrel.item(),
                "LR/lr": scheduler.get_last_lr()[0],
            }, step=epoch, commit=True)

        # -----------------------
        # checkpoint
        # -----------------------
        if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=opt.workspace)

            if accelerator.is_main_process:
                # 用 unwrap_model 更稳（无论是否 DDP/FSdp）
                unwrapped = accelerator.unwrap_model(model)
                # 兼容 optimizer/scheduler 两种常见结构
                opt_state = optimizer.state_dict()
                sched_obj = getattr(scheduler, "scheduler", scheduler)
                sched_state = sched_obj.state_dict()
                checkpoint = {
                    'model': unwrapped.state_dict(),
                    'optimizer': opt_state,
                    'scheduler': sched_state,
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(opt.workspace, f'checkpoint_ep{epoch:03d}.pth'))

        # -----------------------
        # eval (radar metrics)
        # -----------------------
        with torch.no_grad():
            model.eval()
            eval_loss = 0.0
            eval_cd = 0.0
            eval_rcs = 0.0
            eval_vrel = 0.0

            for i, data in enumerate(test_dataloader):
                out = model(data, epoch, vis=0)
                eval_loss += out['loss'].detach()
                eval_cd += out.get('loss_cd', torch.tensor(0.0, device=out['loss'].device)).detach()
                eval_rcs += out.get('loss_rcs', torch.tensor(0.0, device=out['loss'].device)).detach()
                eval_vrel += out.get('loss_vrel', torch.tensor(0.0, device=out['loss'].device)).detach()

            eval_loss = accelerator.gather_for_metrics(eval_loss).mean()
            eval_cd = accelerator.gather_for_metrics(eval_cd).mean()
            eval_rcs = accelerator.gather_for_metrics(eval_rcs).mean()
            eval_vrel = accelerator.gather_for_metrics(eval_vrel).mean()

            if accelerator.is_main_process:
                num_eval_batches = max(1, len(test_dataloader))
                eval_loss /= num_eval_batches
                eval_cd /= num_eval_batches
                eval_rcs /= num_eval_batches
                eval_vrel /= num_eval_batches

                writer.add_scalar("Loss/eval_total", eval_loss, epoch)
                writer.add_scalar("Loss/eval_cd", eval_cd, epoch)
                writer.add_scalar("Loss/eval_rcs", eval_rcs, epoch)
                writer.add_scalar("Loss/eval_vrel", eval_vrel, epoch)

                wandb.log({
                    "Loss/eval_total": eval_loss.item(),
                    "Loss/eval_cd": eval_cd.item(),
                    "Loss/eval_rcs": eval_rcs.item(),
                    "Loss/eval_vrel": eval_vrel.item(),
                }, step=epoch, commit=True)


if __name__ == "__main__":
    main()
