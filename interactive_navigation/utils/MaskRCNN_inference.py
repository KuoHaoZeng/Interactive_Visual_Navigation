import torch
from interactive_navigation.utils.train_mask_rcnn import get_model_instance_segmentation
from PIL import Image
import numpy as np
import cv2

if __name__ == "__main__":
    root_dir = "storage/heuristic_corner_detector_test_scene"
    OBJ = sorted(["ArmChair", "DogBed", "Box", "Chair", "Desk", "DiningTable", "SideTable", "Sofa",
                  "Stool", "Television", "Pillow", "Bread", "Apple", "AlarmClock", "Lettuce",
                  "GarbageCan", "Laptop", "Microwave", "Pot", "Tomato"])
    SIZE = (400, 400)
    COLOR = [(0, 0, 0),
             (255, 0, 0),
             (255, 128, 0),
             (255, 255, 0),
             (128, 255, 0),
             (0, 255, 0),
             (0, 255, 128),
             (0, 255, 255),
             (0, 128, 255),
             (0, 0, 255),
             (127, 0, 255),
             (255, 0, 255),
             (255, 0, 127),
             (128, 128, 128),
             (102, 102, 0),
             (102, 0, 0),
             (0, 102, 0),
             (0, 102, 102),
             (0, 0, 102),
             (102, 0, 102),
             (255, 255, 255)]

    gpu_id = 0
    loc = 'cuda:{}'.format(gpu_id)
    checkpoint = torch.load("storage/maskRcnn/model_9.pth", map_location=loc)
    model = get_model_instance_segmentation(21).cuda(gpu_id)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    imgs = []
    for obj in OBJ:
        img = Image.open("{}/{}/obs.png".format(root_dir, obj))
        img = img.resize(SIZE)
        img = np.array(img) / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img).to(gpu_id)
        imgs.append(img)
    output = model(imgs)
    for j, out in enumerate(output):
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        masks = out["masks"].squeeze().detach().cpu().numpy()
        result = np.zeros((SIZE[0], SIZE[1], 3))
        for i, label in enumerate(labels):
            if label == (j + 1):
                if scores[i] > 0.5:
                    color = COLOR[label]
                    mask = masks[i]
                    idx = np.where(mask > 0.1)
                    result[idx] = color
        result = result.astype(np.uint8)
        img = Image.fromarray(result)
        img.save("{}/{}/maskrcnn_segmentation.png".format(root_dir, OBJ[j]))

        org_img = np.transpose(imgs[j].detach().cpu().numpy() * 255, (1, 2, 0)).astype(np.uint8)
        fin = cv2.addWeighted(result, 0.5, org_img, 0.85, 0)
        fin = Image.fromarray(fin)
        fin.save("{}/{}/maskrcnn_segmentation_blend.png".format(root_dir, OBJ[j]))

        org_img = Image.fromarray(org_img)
        org_img.save("{}/{}/maskrcnn_org_image.png".format(root_dir, OBJ[j]))

    xxx = 0