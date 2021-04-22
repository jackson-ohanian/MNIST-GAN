import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

############################################################################
# Below is my application of a traditional GAN to the MNIST dataset

# Credit for data to MNIST
# LeCun, Y. & Cortes, C. (2010). MNIST handwritten digit database. , .

from load_MNIST import BASE_URL, download, GANDataset

############################################################################
### class DscNet()
### nn.Module()
###
## class DscNet() : A Discriminator Neural Net designed to determine if an
## image belongs to MNIST
############################################################################

class DscNet(nn.Module):
    """Discriminator Player Net"""

    def __init__(self):
        super(DscNet, self).__init__()

        self.layers = nn.Sequential (
            nn.Conv2d(in_channels = 1, out_channels=2, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 2, out_channels=4, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 4, out_channels=8, kernel_size = 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features = 200, out_features = 1)
        )

        self._weight_init()

    def _weight_init(self):
    
        # initializing weights as kaiming uniform distributions
        def init_weights(layer):
            if hasattr(layer, "weight"):
                torch.nn.init.kaiming_uniform_(layer.weight.data)
                layer.bias.data.fill_(0.0)

        self.layers.apply(init_weights)
        return 0

    def forward(self, x):
        return self.layers(x)
        

############################################################################
### class GenNet()
### nn.Module()
###
## class GenNet() : A Generator Neural Net, designed to mimick MNIST images.
############################################################################

class GenNet(nn.Module):
    """Generator Player Net"""

    def __init__(self, zdim):

        super(GenNet, self).__init__()

        self.linear = nn.Linear(in_features = zdim, out_features = 1568)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.layers = torch.nn.Sequential (
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels = 32, out_channels=16, kernel_size = 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels = 16, out_channels=8, kernel_size = 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = 8, out_channels=1, kernel_size = 3, stride=1, padding=1),
            nn.Sigmoid()
            )
        # TODO: implement layers here

        self._weight_init()

    def _weight_init(self):
    
        # initializing weights as kaiming uniform distributions
        torch.nn.init.kaiming_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)

        def init_weights(layer):
            if hasattr(layer, "weight"):
                torch.nn.init.kaiming_uniform_(layer.weight.data)
                layer.bias.data.fill_(0.0)

            self.layers.apply(init_weights)
        return 0

    def forward(self, z):

        z = self.linear(z)
        z = self.lrelu1(z)
        z = z.view(-1, 32, 7, 7)
        z = self.layers(z)

        return z


############################################################################
### class GAN
## param @ self
## param @ zdim : The size of the latent dimension
###
## class GAN : initiate an Adversial Network between a discriminator and
## a generator.
############################################################################

class GAN:
    
    def __init__(self, zdim=64):

        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DscNet().to(self._dev)
        self.gen = GenNet(self._zdim).to(self._dev)


    ############################################################################
    ### _get_loss_d()
    ## param @ self
    ## param @ batch_size : The batch size of real images
    ## param @ batch_data : a batch of real images
    ## param @ z : The latent variable used to discriminate
    ## return @ loss experienced by the discriminator
    ###
    ## _get_loss_d() : Get generator loss (BCE) per epoch, called from train().
    ############################################################################
    
    def _get_loss_d(self, batch_size, batch_data, z):

        smooth = 0.5

        loss = torch.nn.BCEWithLogitsLoss()
        fake_imgs = self.gen(z)
        all_imgs = torch.cat((batch_data, fake_imgs), 0)
        try1 = self.disc(all_imgs)

        # In the optimal case, all generated images are discriminated as false (0)
        target = torch.cat((torch.ones(batch_size, 1), torch.zeros(batch_size, 1)), 0)

        loss_d = smooth*loss(try1, target)

        return loss_d

    ############################################################################
    ### _get_loss_g()
    ## param @ self
    ## param @ batch_size : The batch size of real images
    ## param @ z : The latent variable used to generate
    ## return @ loss experienced by the generator
    ###
    ## _get_loss_g() : Get generator loss (BCE) per epoch, called from train().
    ############################################################################
    
    def _get_loss_g(self, batch_size, z):

        loss = torch.nn.BCEWithLogitsLoss()
        loss_on_gen = self.disc(self.gen(z))

        # In the optimal case, Disc classifies every generated image as real (1)
        target = torch.ones(batch_size, 1)

        loss_g = loss(loss_on_gen, target)
        return loss_g


    ############################################################################
    ### train()
    ## param @ self
    ## param @ iter_d : iter discriminator loss per epoch
    ## param @ iter_g : iter generator loss per epoch
    ## param @ n_epochs : number of epochs of training to perform
    ## param @ batch_size : number of real images per batch (dsc/gen)
    ## param @ lr : learning rate for optimizer(s)
    ## return @ none
    ###
    ## train() : train a General Adversial Network on MNIST image data. Prints
    ##      generator outputs at multiples of 100 epochs.
    ############################################################################

    def train(self, iter_d=1, iter_g=1, n_epochs=200, batch_size=256, lr=0.0002):
        print("----- start training -----")

       ### Not Uploading MNIST / add appropriate file paths / parsing here
       ### if working off of this 
       
        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.SGD(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch % 100 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                # perform step on discriminator
                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                # perform step on generator
                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")

if __name__ == "__main__":
    gan = GAN()
    gan.train()
