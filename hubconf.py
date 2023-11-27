python


class PyTorchHubModel:
    def __init__(self):
        dependencies = ['torch', 'yaml']
        check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('pycocotools', 'thop'))
        set_logging()

    def create(self, name, pretrained, channels, classes, autoshape):
        try:
            cfg = list((Path(__file__).parent / 'cfg').rglob(f'{name}.yaml'))[0]  # model.yaml path
            model = Model(cfg, channels, classes)
            if pretrained:
                fname = f'{name}.pt'  # checkpoint filename
                attempt_download(fname)  # download if not found locally
                ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
                msd = model.state_dict()  # model state_dict
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
                if autoshape:
                    model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
            device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
            return model.to(device)

        except Exception as e:
            s = 'Cache maybe be out of date, try force_reload=True.'
            raise Exception(s) from e

    def custom(self, path_or_model='path/to/model.pt', autoshape=True):
        model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
        if isinstance(model, dict):
            model = model['ema' if model.get('ema') else 'model']  # load model

        hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
        hub_model.load_state_dict(model.float().state_dict())  # load state_dict
        hub_model.names = model.names  # class names
        if autoshape:
            hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
        return hub_model.to(device)

    def yolov7(self, pretrained=True, channels=3, classes=80, autoshape=True):
        return self.create('yolov7', pretrained, channels, classes, autoshape)

if __name__ == '__main__':
    model = PyTorchHubModel()
    hub_model = model.custom(path_or_model='yolov7.pt')  # custom example
    # hub_model = model.create(name='yolov7', pretrained=True, channels=3, classes=80, autoshape=True)  # pretrained example

    # Verify inference
    import numpy as np
    from PIL import Image

    imgs = [np.zeros((640, 480, 3))]

    results = hub_model(imgs)  # batched inference
    results.print()
    results.save()
