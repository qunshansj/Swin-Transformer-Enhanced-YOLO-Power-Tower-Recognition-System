python


class ModelExporter:
    def __init__(self, weights_path, img_size, batch_size, dynamic, dynamic_batch, grid, end2end, max_wh, topk_all, iou_thres, conf_thres, device, simplify, include_nms, fp16, int8):
        self.weights_path = weights_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.dynamic = dynamic
        self.dynamic_batch = dynamic_batch
        self.grid = grid
        self.end2end = end2end
        self.max_wh = max_wh
        self.topk_all = topk_all
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.device = device
        self.simplify = simplify
        self.include_nms = include_nms
        self.fp16 = fp16
        self.int8 = int8

    def export(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default=self.weights_path, help='weights path')
        parser.add_argument('--img-size', nargs='+', type=int, default=self.img_size, help='image size')  # height, width
        parser.add_argument('--batch-size', type=int, default=self.batch_size, help='batch size')
        parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
        parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
        parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
        parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
        parser.add_argument('--max-wh', type=int, default=self.max_wh, help='None for tensorrt nms, int value for onnx-runtime nms')
        parser.add_argument('--topk-all', type=int, default=self.topk_all, help='topk objects for every images')
        parser.add_argument('--iou-thres', type=float, default=self.iou_thres, help='iou threshold for NMS')
        parser.add_argument('--conf-thres', type=float, default=self.conf_thres, help='conf threshold for NMS')
        parser.add_argument('--device', default=self.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
        parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
        parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
        parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
        opt = parser.parse_args()
        opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
        opt.dynamic = opt.dynamic and not opt.end2end
        opt.dynamic = False if opt.dynamic_batch else opt.dynamic
        print(opt)
        set_logging()
        t = time.time()

        # Load PyTorch model
        device = select_device(opt.device)
        model = attempt_load(opt.weights, map_location=device)  # load FP32 model
        labels = model.names

        # Checks
        gs = int(max(model.stride))  # grid size (max stride)
        opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

        # Input
        img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

        # Update model
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, models.common.Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            # elif isinstance(m, models.yolo.Detect):
            #     m.forward = m.forward_export  # assign forward (optional)
        model.model[-1].export = not opt.grid  # set Detect() layer grid export
        y = model(img)  # dry run
        if opt.include_nms:
            model.model[-1].include_nms = True
            y = None

        # TorchScript export
        try:
            print('\nStarting TorchScript export with torch %s...' % torch.__version__)
            f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
            ts = torch.jit.trace(model, img, strict=False)
            ts.save(f)
            print('TorchScript export success, saved as %s' % f)
        except Exception as e:
            print('TorchScript export failure: %s' % e)

        # CoreML export
        try:
            import coremltools as ct

            print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
            # convert model from torchscript and apply pixel scaling as per detect.py
            ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
            bits, mode = (8, 'kmeans_lut') if opt.int8 else (16, 'linear') if opt.fp16 else (32, None)
            if bits < 32:
                if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                        ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
                else:
                    print('quantization only supported on macOS, skipping...')

            f = opt.weights.replace('.pt', '.mlmodel')  # filename
            ct_model.save(f)
            print('CoreML export success, saved as %s' % f)
        except Exception as e:
            print('CoreML export failure: %s' % e)

        # TorchScript-Lite export
        try:
            print('\nStarting TorchScript-Lite export with torch %s...' % torch.__version__)
            f = opt.weights.replace('.pt', '.torchscript.ptl')  # filename
            tsl = torch.jit.trace(model, img, strict=False)
            tsl = optimize_for_mobile(tsl)
            tsl._save_for_lite_interpreter(f)
            print('TorchScript-Lite export success, saved as %s' % f)
        except Exception as e:
            print('TorchScript-Lite export failure: %s' % e)

        # ONNX export
        try:
            import onnx

            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            f = opt.weights.replace('.pt', '.onnx')  # filename
            model.eval()
            output_names = ['classes', 'boxes'] if y is None else ['output']
            dynamic_axes = None
            if opt.dynamic:
                dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                 'output': {0: 'batch', 2: 'y', 3: 'x'}}
            if opt.dynamic_batch:
                opt.batch_size = 'batch'
                dynamic_axes = {
                    'images': {
                        0: 'batch',
                    }, }
                if opt.end2end and opt.max_wh is None:
                    output_axes = {
                       
