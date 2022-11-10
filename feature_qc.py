import os

from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
import numpy as np

from dataloaders import DModule

def main():

    img_size = 448
    folder_path = os.path.normpath("..//..//sea_ice//data//aerial_kaggle//dataset//semantic_drone_dataset")
    dir_out = os.path.normpath("..//..//sea_ice//data//aerial_kaggle/experiments")

    # the images are resized to 'resize_to'. 
    # the primary image is obtained by the center crop of the resized image
    # the secondary image is obtained from a random crop of size out_size
    # extracted from the resized image. 
    out_size = 448 # output size 
    resize_to = 800 # rescale the images to this number of pixels before cropping

    n_total_samples = 400 #400 is the total number of samples available
    n_val_samples = 64 # n_val_samples will be selected from n_total_samples

    batch_size = 4

    num_workers = 12

    ############################################################################
    weights=ResNet50_Weights.IMAGENET1K_V2
    backbone = resnet50(replace_stride_with_dilation=[False, True, True], 
                                    weights=weights)
    return_layers = {"layer4": "out"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)                                

    backbone.eval()

    # create datamodule:
    dm = DModule(folder_path=folder_path, 
                n_training_samples=n_total_samples, 
                n_val_samples=n_val_samples,
                out_size=out_size,
                resize_to=resize_to,
                batch_size=batch_size,
                num_workers=num_workers
                )

    # run setup to get split
    dm.setup(None)

    print('Setup complete.')

    mnist = MNIST(
        root = dir_out,
        train = True,
        transform = transforms.ToTensor(), 
        download = True,            
    )

    resize = transforms.Resize(size=img_size)

    next(iter(mnist))
    mnist = resize(next(iter(mnist))[0])
    mnist=torch.concat([mnist, mnist, mnist], 0)
    mnist = torch.unsqueeze(mnist, 0)


    for imgs_b, tar_b in dm.train_dataloader():
        print(imgs_b[0].shape)
        break

    mnist = torch.unsqueeze(imgs_b[0][0], 0)

    for shift in range(5, img_size, 5):
        print(f'shifting by {shift}')

        _temp = torch.clone(mnist)
        # move
        _temp[:,:,shift:,:] = mnist[:,:,:-shift,:]
        _temp[:,:,:shift,:] = mnist[:,:,-shift:,:]

        fig, ax = plt.subplots(ncols=3, figsize=(12,4))

        # original
        _temp_show = torch.squeeze(mnist)
        _temp_show = np.swapaxes(_temp_show.numpy(),0,-1)[0:,:,:]
        ax[0].imshow(_temp_show)
        centers = np.array(_temp_show.shape)/2
        ax[0].scatter([centers[1]], [centers[1]], c='red')

        # shift
        _temp_show = torch.squeeze(_temp)
        _temp_show = np.swapaxes(_temp_show.numpy(),0,-1)[0:,:,:]
        ax[1].imshow(_temp_show)

        centers = np.array(_temp_show.shape)/2
        ax[1].scatter([centers[1]], [centers[1]], c='red')

        # results
        res = backbone(_temp)['out']
        _temp_show = torch.squeeze(res.detach())
        _temp_show = np.swapaxes(_temp_show.numpy(),0,-1)
        ax[2].imshow(_temp_show.mean(-1))

        centers = np.array(_temp_show.shape)/2
        ax[2].scatter([centers[1]], [centers[1]], c='red')


        fig.suptitle(f'image size: {img_size} shif: {shift}')

        fig.savefig(os.path.join(dir_out, f'mnist-{shift}'))
        plt.close('all')

######################################temp qc
        # for sample in range(0, 9, 3):

        #     img_center = torch.unsqueeze(imgs_b[0][sample,:,:], 0)
        #     for shift in range(5, out_size, 10):
        #         print(f'shifting by {shift}')

        #         _temp = torch.clone(img_center)
        #         # move
        #         _temp[:,:,shift:,:] = img_center[:,:,:-shift,:]
        #         _temp[:,:,:shift,:] = img_center[:,:,-shift:,:]

        #         fig, ax = plt.subplots(ncols=4, figsize=(12,4))

        #         # original
        #         _temp_show = torch.squeeze(img_center)
        #         _temp_show = np.swapaxes(_temp_show.numpy(),0,-1)[:,:,0]
        #         ax[0].imshow(_temp_show)
        #         centers = np.array(_temp_show.shape)/2
        #         ax[0].scatter([centers[1]], [centers[1]], c='red')
        #         ax[0].set_title('original')

        #         # shift
        #         _temp_show = torch.squeeze(_temp)
        #         _temp_show = np.swapaxes(_temp_show.numpy(),0,-1)[:,:,0]
        #         ax[1].imshow(_temp_show)

        #         centers = np.array(_temp_show.shape)/2
        #         ax[1].scatter([centers[1]], [centers[1]], c='red')
        #         ax[1].set_title(f'shifted by {shift} pixels')

        #         # results
        #         idx = 2
        #         for backbone in model.streams:
        #             backbone.eval()
        #             res = backbone(_temp)['out']
        #             _temp_show = torch.squeeze(res.detach())
        #             _temp_show = np.swapaxes(_temp_show.numpy(),0,-1)
        #             ax[idx].imshow(_temp_show.mean(-1))

        #             centers = np.array(_temp_show.shape)/2
        #             ax[idx].scatter([centers[1]], [centers[1]], c='red')
                    
        #             ax[idx].set_title(f'shifted through stream {idx-1}')

        #             idx+=1


        #         fig.suptitle(f'image size: {out_size} shif: {shift}')

        #         fig.savefig(os.path.join(dir_out, f'{sample}-shift-check-{shift}'))
        #         plt.close('all')



if __name__ == '__main__':
    main()



