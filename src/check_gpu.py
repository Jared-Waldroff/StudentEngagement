import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print("Torchvision version:", torchvision.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Test torchvision.ops.nms on CUDA
    try:
        boxes = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]], dtype=torch.float32).cuda()
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
        iou_threshold = 0.5

        selected_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        print("Selected indices:", selected_indices)
    except Exception as e:
        print("Error during NMS:", e)
else:
    print("CUDA is not available.")
    print("CUDA is not available.")
