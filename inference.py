import time
import cv2
import numpy as np
import onnxruntime

from utils import xywh2xyxy, draw_detections, multiclass_nms

class TargetDetection:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        print(self.input_width,self.input_height)

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

# yolov8 onnx 模型推理
class ATDetector():
    def __init__(self):
        super(ATDetector, self).__init__()
        self.model_path = "./Jjianbest.onnx"
        self.detector = TargetDetection(self.model_path, conf_thres=0.5, iou_thres=0.3)

    def detect_image(self, input_image, output_image):
        cv_img = cv2.imread(input_image)
        boxes, scores, class_ids = self.detector.detect_objects(cv_img)
        cv_img = draw_detections(cv_img, boxes, scores, class_ids)
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imwrite(output_image, cv_img)
        # cv2.imshow('output', cv_img)
        # cv2.waitKey(0)

    def detect_video(self, input_video, output_video):
        cap = cv2.VideoCapture(input_video)
        fps = int(cap.get(5))
        videoWriter = None

        while True:
            _, cv_img = cap.read()
            if cv_img is None:
                break
            boxes, scores, class_ids = self.detector.detect_objects(cv_img)
            cv_img = draw_detections(cv_img, boxes, scores, class_ids)

            # 如果视频写入器未初始化，则使用输出视频路径和参数进行初始化
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                # 在这里给值了，它就不是None, 下次判断它就不进这里了
                videoWriter = cv2.VideoWriter(output_video, fourcc, fps, (cv_img.shape[1], cv_img.shape[0]))

            videoWriter.write(cv_img)
            cv2.imshow("aod", cv_img)
            cv2.waitKey(5)

            # 等待按键并检查窗口是否关闭
            if cv2.getWindowProperty("aod", cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    det = ATDetector()
    input_image = "./test_image.jpg"
    output_image = './output.jpg'
    det.detect_image(input_image, output_image)


