import torch
import os


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_acc,
    checkpoint_dir,
    filename='best_model.pt',
    scheduler=None,
):
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None):
    device = next(model.parameters()).device
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    return epoch, best_acc
