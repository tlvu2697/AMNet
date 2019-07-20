import math
import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from amnet_model import *
import amnet_model as amnet_model

class AMNet:
    def __init__(self):
        self.model = None
        self.transform = None
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.target_mean = 0.754
        self.target_scale = 2.0

        # self.cnn_core = 'ResNet50FC'
        # self.checkpoint = 'assessors/lamem_ResNet50FC_lstm3_train_5_weights_35.pkl'
        self.cnn_core = 'ResNet101FC'
        self.checkpoint = 'assessors/lamem_ResNet101FC_lstm3_train_5_weights_30.pkl'
        return

    def init(self):
        core_cnn = getattr(amnet_model, self.cnn_core)()
        model = AMemNetModel(core_cnn, None, a_res=14, a_vec_size=1024)

        rnd_seed = 12345
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        torch.cuda.set_device(0)
        torch.cuda.manual_seed(rnd_seed)

        self.model = model
        self.init_transformations()
        self.load_checkpoint(self.checkpoint)
        return

    def init_transformations(self):
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.img_mean, std=self.img_std)
        ])
        return

    def load_checkpoint(self, filename):
        if filename.strip() == '':
            return False

        try:
            print('Loading checkpoint: ', filename)
            cpnt = torch.load(filename, map_location=lambda storage, loc: storage)
        except FileNotFoundError:
            print("Cannot open file: ", filename)
            self.model_weights_current = ''
            return False

        self.model.load_state_dict(cpnt['model'], strict=False)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.cuda()

        return True

    def postprocess(self, outputs):
        output = (outputs).sum(1)
        output = output / outputs.shape[1]

        output /= self.target_scale
        output = output + self.target_mean

        return output

    def predict(self, data):

        _, outputs_, _ = self.model(data)
        memity = self.postprocess(outputs_)

        return memity

def _tencrop_image_transform(model):
    normalize = torchvision.transforms.Normalize(mean=model.img_mean, std=model.img_std)
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda image: _tencrop(image.permute(0, 2, 3, 1), cropped_size=224)),
        torchvision.transforms.Lambda(lambda image: torch.stack([torch.stack([normalize(x / 255) for x in crop])
                                                                 for crop in image])),
    ])

def _tencrop_output_transform_emonet(output):
    output = output.view(-1, 10).mean(1)
    return output

def _image_transform(model):
    normalize = torchvision.transforms.Normalize(mean=model.img_mean, std=model.img_std)
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(224, 224), mode="bilinear")),
        torchvision.transforms.Lambda(lambda image: torch.stack([normalize(x / 255) for x in image])),
    ])


def _tencrop(images, cropped_size=227):
    im_size = 256  # hard coded

    crops = torch.zeros(images.shape[0], 10, 3, cropped_size, cropped_size)
    indices = [0, im_size - cropped_size]  # image size - crop size

    for img_index in range(images.shape[0]):  # looping over the batch dimension
        img = images[img_index, :, :, :]
        curr = 0
        for i in indices:
            for j in indices:
                temp_img = img[i:i + cropped_size, j:j + cropped_size, :]
                crops[img_index, curr, :, :, :] = temp_img.permute(2, 0, 1)
                crops[img_index, curr + 5, :, :, :] = torch.flip(crops[img_index, curr, :, :, :], [2])
                curr = curr + 1
        center = int(math.floor(indices[1] / 2) + 1)
        crops[img_index, 4, :, :, :] = img[center:center + cropped_size,
                                           center:center + cropped_size, :].permute(2, 0, 1)
        crops[img_index, 9, :, :, :] = torch.flip(crops[img_index, curr, :, :, :], [2])
    return crops

def amnet(tencrop):
    model = AMNet()
    model.init()

    if tencrop:
        input_transform = _tencrop_image_transform(model)
        output_transform = _tencrop_output_transform_emonet
    else:
        input_transform = _image_transform(model)
        output_transform = lambda x: x

    return model.predict, input_transform, output_transform
