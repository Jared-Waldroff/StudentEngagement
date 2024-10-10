# save_model.py
import torch


def save_model(model, path='daisee_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
