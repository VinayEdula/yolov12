from ultralytics import YOLO

def train_yolov12_pcb(model_size='n'):
    model = YOLO('yolov12m.pt')
    data_config = "defects.yaml"
    print(f"Initializing YOLOv12{model_size.upper()} model...")

    training_args = {
        'data': data_config,
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'device': 0,  
        'workers': 8,
        'patience': 50,
        'save': True,
        'save_period': 10,  
        'cache': False,
        'project': 'runs/detect',
        'name': f'pcb_defects_yolov12{model_size}',  
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'plots': True
    }
    
    print("Starting training...")
    print(f"Dataset: {data_config}")
    print(f"Model: YOLOv12{model_size.upper()}")
    print(f"Classes: 6 (missing_hole, open_circuit, short, spur, spurious_copper, mouse_bite)")
    
    # Start training
    results = model.train(**training_args)
    return results



model_size = 'n'  
print(f"Training YOLOv12{model_size.upper()} for PCB defect detection...")
results = train_yolov12_pcb(model_size=model_size)

if results:
    print(f"\nTraining successful! Best weights saved at:")
    print(f"{results.save_dir}/weights/best.pt")
    print(f"\nTo use the trained model:")
    print(f"model = YOLO('{results.save_dir}/weights/best.pt')")
    print(f"results = model('path/to/image.jpg')")