import sys
import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, BASE_PATH)

from torchvision import transforms
from models.ssd.utils import *
from utils.utils import *
from PIL import Image, ImageDraw, ImageFont
from utils.datasets import SingleImage



# # Load model checkpoint
# checkpoint = '/home/salvacarrion/Documents/Programming/Python/Projects/yolo4math/models/ssd/BEST_checkpoint_ssd300.pth.tar'
# checkpoint = torch.load(checkpoint)
# start_epoch = checkpoint['epoch'] + 1
# best_loss = checkpoint['best_loss']
# print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
# model = checkpoint['model']
# model = model.to(device)

# Transforms
resize = transforms.Resize((1024, 1024))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(model, image_path, min_score, max_overlap, top_k, suppress=None, class_names=None, save_path=None,
           input_size=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    #image = normalize(to_tensor(resize(original_image)))
    tf = SingleImage(input_size)
    image = tf.apply_transform(image_path)
    # image = normalize(image1)

    # Move to default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # Forward prop.
    model.eval()
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k, cpu=True)

    # Clip predictions
    for i in range(len(det_boxes)):
        det_boxes[i] = torch.clamp(det_boxes[i], min=0.0, max=1.0)

    p_bboxes = det_boxes[0]
    p_labels = det_labels[0]
    plot_bboxes(image, p_bboxes, class_ids=p_labels, class_names=class_names, show_results=False,
                title="Detection", coords_rel=True, save_path=save_path)

    return None


if __name__ == '__main__':
    model_path = "/home/salvacarrion/Documents/Programming/Python/Projects/yolo4math/checkpoints/ssd_best.pth"
    model = torch.load(model_path)
    model.eval()

    img_path = '/home/salvacarrion/Documents/datasets/equations/1024/{}'
    class_names = ['background', 'embedded', 'isolated']
    for p in ["10.1.1.1.2018_5.jpg", "10.1.1.1.2007_11.jpg", "10.1.1.1.2005_13.jpg"]:
        p = img_path.format(p)
        detect(model, p, min_score=0.2, max_overlap=0.3, top_k=200, input_size=(512, 400), class_names=class_names)

    #
    # img_path = '/home/salvacarrion/Documents/datasets/VOC2007/JPEGImages/{}'
    #
    # class_names = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
    #                "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14,
    #                "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
    # class_names = list(class_names.keys())
    #
    # for p in ['008344.jpg', '006324.jpg', '0008843.jpg', '008344.jpg', '001262.jpg']:
    #     p = img_path.format(p)
    #     detect(model, p, min_score=0.5, max_overlap=0.3, top_k=200, input_size=(300, 300), class_names=class_names)
    #     ads = 33
