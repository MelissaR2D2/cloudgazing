import torch
import matplotlib.pyplot as plt
from dataloader import SWIMGSEG
from image_display_util import create_cloud_mask, create_segmented_image
from train import decide
# Given an arbitrary cloud image, produces image outlining cloud and object

dataset_dir = "/Users/student/Documents/College/swimseg_split/test/images/0/"
results_dir = "/Users/student/PycharmProjects/CS474FinalProject/results/"
cloud_model_path = "UNetCloudBase.pt"
object_model_path = "UNetVOCBase.pt"

dataset = SWIMGSEG(train=False, image_size=256)
org_image, _ = dataset.__getitem__(12)
org_image = org_image

cloud_model = torch.load(results_dir + cloud_model_path, map_location=torch.device('cpu'))
cloud_result = decide(cloud_model(org_image.unsqueeze(0))).squeeze()

masked_image = create_cloud_mask(org_image, cloud_result)

# plt.imshow(masked_image.permute(1, 2, 0))
# plt.show()

object_model = torch.load(results_dir + object_model_path, map_location=torch.device('cpu'))
object_result = decide(object_model(masked_image.unsqueeze(0))).squeeze()

final_image = create_segmented_image(org_image, object_result)
plt.imshow(final_image)
plt.show()
