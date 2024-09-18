# dataset.py
import deeplake

def load_daisee_datasets():
    # Load training, validation, and testing datasets
    train_ds = deeplake.load("hub://activeloop/daisee-train")
    val_ds = deeplake.load("hub://activeloop/daisee-validation")
    test_ds = deeplake.load("hub://activeloop/daisee-test")

    # Convert datasets to PyTorch Dataloaders
    train_loader = train_ds.pytorch(num_workers=0, batch_size=4, shuffle=True)
    val_loader = val_ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
    test_loader = test_ds.pytorch(num_workers=0, batch_size=4, shuffle=False)

    return train_loader, val_loader, test_loader
