# main.py
from dataset import load_daisee_datasets
from train import train_model
from save_model import save_model


def main():
    # Load the DAiSEE dataset
    train_loader, val_loader, test_loader = load_daisee_datasets()

    # Train the model
    model = train_model(train_loader, val_loader, num_epochs=5, learning_rate=0.001)

    # Save the model
    save_model(model, path='daisee_model.pth')


if __name__ == '__main__':
    main()
