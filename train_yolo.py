from ultralytics import YOLO
yaml_path: str = './datasets/infrare_dataset/data.yaml'
def train(dataset_yaml_path: str = None):
    model = YOLO('yolov8s.pt').to('cuda')
    model.train(task='detect', mode='train', data=dataset_yaml_path, imgsz=640, batch=8, epochs=100, lr0=0.002)
if __name__ == "__main__":
    train(yaml_path)