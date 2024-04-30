import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from torch.utils import data
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
import torch.autograd as autograd


# Encoder
class DeterministicConditional(nn.Module):
    def __init__(self, args, shift=None):
        super(DeterministicConditional, self).__init__()

        self.encoder = nn.Sequential(
            Conv2d(3, args.DIM, 4, 2, 1, bias=False), BatchNorm2d(args.DIM), ReLU(inplace=True),
            Conv2d(args.DIM, args.DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(args.DIM * 2), ReLU(inplace=True),
            Conv2d(args.DIM * 2, args.DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(args.DIM * 4), ReLU(inplace=True),
            Conv2d(args.DIM * 4, args.DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(args.DIM * 4), ReLU(inplace=True),
            Conv2d(args.DIM * 4, 100, 1, 1, 0))

        self.shift = shift

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, image):
        output = self.encoder(image)
        if self.shift is not None:
            output = output + self.shift
        return output


class Generator(nn.Module):
    def __init__(self, args, shift=None):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            ConvTranspose2d(100, args.DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(args.DIM * 4), ReLU(inplace=True),
            ConvTranspose2d(args.DIM * 4, args.DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(args.DIM * 2),
            ReLU(inplace=True),
            ConvTranspose2d(args.DIM * 2, args.DIM, 4, 2, 1, bias=False), BatchNorm2d(args.DIM), ReLU(inplace=True),
            ConvTranspose2d(args.DIM, 3, 4, 2, 1, bias=False), Tanh())

        self.shift = shift

    def set_shift(self, value):
        if self.shift is None:
            return
        assert list(self.shift.data.size()) == list(value.size())
        self.shift.data = value

    def forward(self, image):
        output = self.generator(image)
        if self.shift is not None:
            output = output + self.shift
        return output


class JointCritic(nn.Module):
    def __init__(self, args):
        super(JointCritic, self).__init__()

        self.x_mapping = nn.Sequential(
            Conv2d(3, args.DIM, 4, 2, 1), LeakyReLU(0.2),
            Conv2d(args.DIM, args.DIM * 2, 4, 2, 1), LeakyReLU(0.2),
            Conv2d(args.DIM * 2, args.DIM * 4, 4, 2, 1), LeakyReLU(0.2),
            Conv2d(args.DIM * 4, args.DIM * 4, 4, 1, 0), LeakyReLU(0.2))

        self.z_mapping = nn.Sequential(
            Conv2d(100, 512, 1, 1, 0), LeakyReLU(0.2),
            Conv2d(512, 512, 1, 1, 0), LeakyReLU(0.2))

        self.joint_mapping = nn.Sequential(
            Conv2d(args.DIM * 4 + 512, 1024, 1, 1, 0), LeakyReLU(0.2),
            Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(0.2),
            Conv2d(1024, 1, 1, 1, 0))

    def forward(self, x, z):
        assert x.size(0) == z.size(0)
        x_out = self.x_mapping(x)
        z_out = self.z_mapping(z)
        joint_input = torch.cat((x_out, z_out), dim=1)
        output = self.joint_mapping(joint_input)
        return output


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)


class WALI(nn.Module):
    def __init__(self, E, G, C):
        """ Adversarially learned inference (a.k.a. bi-directional GAN) with Wasserstein critic.

    Args:
      E: Encoder p(z|x).
      G: Generator p(x|z).
      C: Wasserstein critic function f(x, z).
    """
        super().__init__()

        self.E = E
        self.G = G
        self.C = C

    def get_encoder_parameters(self):
        return self.E.parameters()

    def get_generator_parameters(self):
        return self.G.parameters()

    def get_critic_parameters(self):
        return self.C.parameters()

    def encode(self, x):
        return self.E(x)

    def generate(self, z):
        return self.G(z)

    def reconstruct(self, x):
        return self.generate(self.encode(x))

    def criticize(self, x, z_hat, x_tilde, z):
        input_x = torch.cat((x, x_tilde), dim=0)
        input_z = torch.cat((z_hat, z), dim=0)
        output = self.C(input_x, input_z)
        data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
        return data_preds, sample_preds

    def calculate_grad_penalty(self, x, z_hat, x_tilde, z):
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1, 1).to(x.device)  # eps ~ Unif[0, 1]
        intp_x = eps * x + (1 - eps) * x_tilde
        intp_z = eps * z_hat + (1 - eps) * z
        intp_x.requires_grad = True
        intp_z.requires_grad = True
        C_intp_loss = self.C(intp_x, intp_z).sum()
        grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
        grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
        grads = torch.cat((grads_x, grads_z), dim=1)
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x, z, lamb=10):
        z_hat, x_tilde = self.encode(x), self.generate(z)
        data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
        EG_loss = torch.mean(data_preds - sample_preds)
        C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
        return C_loss, EG_loss


def log_odds(p):
    p = torch.clamp(p.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(p / (1 - p))


class TrainerBiGAN:
    def __init__(self, args, data, device, current_exp=0):

        self.G = Generator(args=args, shift=None)
        self.E = DeterministicConditional(args=args)
        self.C = JointCritic(args=args)

        self.G.apply(weights_init_normal)
        self.E.apply(weights_init_normal)
        self.C.apply(weights_init_normal)

        self.args = args
        self.train_loader = data
        self.device = device
        self.current_exp = current_exp

        self.wali = WALI(self.E, self.G, self.C).to(self.args.device)

        self.optimizer_EG = torch.optim.Adam(list(self.wali.get_encoder_parameters()) +
                                             list(self.wali.get_generator_parameters()),
                                             lr=args.lr, betas=args.betas)

        self.optimizer_C = torch.optim.Adam(self.wali.get_critic_parameters(),
                                            lr=args.lr, betas=args.betas)

        self.noise = torch.randn(64, 100, 1, 1, device=args.device)

        self.C_update = True
        self.EG_update = False

        self.c_losses = []
        self.eg_losses = []

    def train(self):
        """Training the BiGAN"""
        curr_iter = C_iter = EG_iter = 0
        total_iters = self.args.ITER * max(self.args.C_ITERS, self.args.EG_ITERS)

        while curr_iter < self.args.ITER:
            for batch_idx, (x, _) in enumerate(self.train_loader, 1):

                x = x.to(self.args.device)

                if curr_iter == 0:
                    init_x = x
                    curr_iter += 1

                z = torch.randn(x.size(0), 100, 1, 1).to(self.args.device)

                c_loss, eg_loss = self.wali(x, z, lamb=self.args.LAMBDA)

                if self.C_update:
                    self.optimizer_C.zero_grad()

                    c_loss.backward()

                    self.c_losses.append(c_loss.item())

                    self.optimizer_C.step()

                    C_iter += 1

                    if C_iter == C_iter:
                        C_iter = 0
                        self.C_update, self.EG_update = False, True

                    continue

                if self.EG_update:
                    self.optimizer_EG.zero_grad()

                    eg_loss.backward()
                    self.eg_losses.append(eg_loss.item())

                    self.optimizer_EG.step()

                    EG_iter += 1

                    if EG_iter == self.args.EG_ITERS:
                        EG_iter = 0
                        self.C_update, self.EG_update = True, False
                        curr_iter += 1
                    else:
                        continue

                if curr_iter % 400 == 0:
                    print('[%d/%d]\tW-distance: %.4f\tC-loss: %.4f'
                          % (curr_iter, self.args.ITER, eg_loss.item(), c_loss.item()))

                if curr_iter % self.args.EG_ITERS - 1 == 0:

                    # plot reconstructed images and samples
                    self.wali.eval()
                    real_x, rect_x = init_x[:32], self.wali.reconstruct(init_x[:32]).detach_()
                    rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1)
                    rect_imgs = rect_imgs.view(64, 3, self.args.IMAGE_SIZE, self.args.IMAGE_SIZE).cpu()
                    genr_imgs = self.wali.generate(self.noise).detach_().cpu()
                    save_path = '/home/luu/projects/cl_selective_nets/results/BiGAN/images/'
                    torchvision.utils.save_image(rect_imgs * 0.5 + 0.5, save_path
                                                 + f'rect{curr_iter}_{self.current_exp}exp.png')

                    torchvision.utils.save_image(genr_imgs * 0.5 + 0.5, save_path
                                                 + f'genr{curr_iter}_{self.current_exp}exp.png')
                    self.wali.train()
