import sys
import numpy as np
import torch
import torch.nn as nn
import scipy
from models.opennet.util_model import UnitPosNegScale, reshape_pad


class OpenNetBase(object):
    def __init__(self, model, x_dim, y_dim, z_dim=6,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 optimizer=None,
                 recon_opt=None,
                 c_opt=None,
                 dist='mean_separation_spread',
                 decision_dist_fc='euclidean',
                 threshold_type='global',
                 dropout=True, keep_prob=0.7, batch_size=128, iterations=5000,
                 ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True,
                 div_loss=False, combined_loss=False, contamination=0.1):

        """

        :param x_dim: dimension of the input
        :param y_dim: number of known classes.
        :param z_dim: the number of latent variables.
        :param x_scale: an input scaling function. Default scale to range of [-1, 1].
                         If none, the input will not be scaled.
        :param x_inverse_scale: reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                         If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape: a function to reshape the input before feeding to the networks input layer.
        :param optimizer: the Optimizer used when updating based on ii-loss.
                            Used when inter_loss and intra_loss are enabled
        :param recon_opt: the Optimizer used when updating based on reconstruction-loss (Not used ii, ii+ce or ce)
        :param c_opt: the Optimizer used when updating based on cross entropy loss.
                        Used for ce and ii+ce modes (i.e. ce_loss is enabled)
        :param dist: ii-loss calculation mode. Only 'mean_separation_spread' should be used.
        :param decision_dist_fc: outlier score distance functions
        :param threshold_type: outlier threshold mode. 'global' appears to give better results.
        :param dropout:
        :param keep_prob:
        :param batch_size:
        :param iterations:
        :param ce_loss: Consider cross entropy loss. When enabled with intra_loss and inter_loss gives (ii+ce) mode
        :param recon_loss: Experimental! avoid enabling them.
        :param inter_loss: Consider inter-class separation. Should be enabled together with intra_loss for (ii-loss)
        :param intra_loss: Consider intra-class spread. Should be enabled together with inter_loss for (ii-loss)
        :param div_loss: Experimental. avoid enabling them.
        :param combined_loss: Experimental. avoid enabling them.
        :param contamination: contamination ratio used for outlier threshold estimation.
        """

        self.c_cov_inv = None
        self.c_cov = None
        self.c_means = None
        self.model = model

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.x_scale = x_scale
        self.x_inverse_scale = x_inverse_scale
        self.x_reshape = x_reshape

        self.z_dim = z_dim

        self.dropout = dropout
        self.is_training = False
        self.keep_prob = keep_prob

        self.contamination = contamination

        self.opt = optimizer
        self.recon_opt = recon_opt
        self.c_opt = c_opt

        self.dist = dist

        self.decision_dist_fn = decision_dist_fc

        self.threshold_type = threshold_type

        self.enable_ce_loss = ce_loss
        self.enable_intra_loss = intra_loss
        self.enable_inter_loss = inter_loss

        self.model_params = ['x_dim', 'y_dim', 'z_dim', 'dropout', 'keep_prob',
                             'contamination', 'decision_dist_fn', 'dist', 'batch_size',
                             'batch_size', 'iterations', 'enable_ce_loss', 'enable_inter_loss',
                             'enable_intra_loss', 'threshold_type']

        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def model_config(self):
        return {field: val for field, val in vars(self).items() if field in self.model_params}

    def bucket_mean(self, data, bucket_ids, num_buckets):
        """
        # Example data:
            data = torch.tensor([6.0, 2.0, 3.0, 4.0, 5.0])
            bucket_ids = torch.tensor([0, 1, 0, 1, 2])
            num_buckets = 3
        :param data:
        :param bucket_ids:
        :param num_buckets:
        :return:
        """
        total = torch.bincount(bucket_ids, weights=data, minlength=num_buckets)
        count = torch.bincount(bucket_ids, minlength=num_buckets)
        return total / count

    def bucket_max(self, data, bucket_ids, num_buckets):
        b_max = torch.bincount(bucket_ids, weights=data, minlength=num_buckets)
        return b_max

    def x_reformat(self, xs):
        """
        Rescale and reshape x if x_scale and x_reshape functions are provided.
        """
        if self.x_scale is not None:
            xs = self.x_scale(xs)
        if self.x_reshape is not None:
            xs = self.x_reshape(xs)
        return xs

    def _next_batch(self, x, y):
        index = np.random.randint(0, high=x.shape[0], size=self.batch_size)
        return x[index], y[index]

    def encoder(self, x, reuse=False):
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on creating a new one.
        Returns:
            latent var z
        """
        pass

    def decoder(self, z, reuse=False):
        """ Decoder Network. Experimental!
        Args:
            :param z - latent variables z.
            :param reuse - whether to reuse old network to create a new one.
        Returns:
            The reconstructed x
        """
        pass

    def sq_difference_from_mean(self, data, class_mean):
        """Calculates the squared difference from class mean."""
        sq_diff_list = []
        for i in range(self.y_dim):
            sq_diff_list.append(
                torch.mean(
                    torch.square(data - class_mean[i]), dim=1
                )
            )

        return torch.stack(sq_diff_list, dim=1)

    def inter_min_intra_max(self, data, labels, class_mean):
        """Calculates intra-class spread as max distance from class means.
        Calculates inter-class separation as the distance between the two closest class means."""
        inter_min, _ = self.inter_separation_intra_spread(data, labels, class_mean)

        sq_diff = self.sq_difference_from_mean(data, class_mean)

        # Do element-wise multiplication with labels to use as a mask
        masked_sq_diff = sq_diff * labels.float()
        intra_max = torch.sum(torch.max(masked_sq_diff, dim=0)[0])

        return intra_max, inter_min

    def inter_intra_diff(self, data, labels, class_mean):
        """
        Calculates the intra-class and inter-class distance
        as the average distance from the class means.
        """
        sq_diff = self.sq_difference_from_mean(data, class_mean)

        inter_intra_sq_diff = self.bucket_mean(sq_diff, labels, 2)
        inter_class_sq_diff = inter_intra_sq_diff[0]
        intra_class_sq_diff = inter_intra_sq_diff[1]
        return intra_class_sq_diff, inter_class_sq_diff

    def inter_separation_intra_spread(self, data, labels, class_mean):
        """
        Calculates intra-class spread as average distance from class means.
        Calculates inter-class separation as the distance between the two closest class means.
        Returns:
        intra-class spread and inter-class separation."""
        intra_class_sq_diff, _ = self.inter_intra_diff(data, labels, class_mean)

        ap_dist = self.all_pair_distance(class_mean)
        dim = class_mean.size(0)
        not_diag_mask = ~torch.eye(dim, dtype=torch.bool)
        inter_separation = torch.min(ap_dist.masked_select(not_diag_mask))
        return intra_class_sq_diff, inter_separation

    def all_pair_distance(self, A):
        r = torch.sum(A * A, dim=1)

        # turn r into column vector
        r = r.view(-1, 1)
        D = r - 2 * torch.matmul(A, A.transpose(0, 1)) + r.transpose(0, 1)
        return D

    def all_pair_inter_intra_diff(self, xs, ys):
        """
        Calculates the intra-class and inter-class distance as the
        average distance between all pair of instances intra and inter class instances.
        """

        def outer(ys):
            return torch.matmul(ys, ys.transpose(0, 1))

        ap_dist = self.all_pair_distance(xs)
        mask = outer(ys)

        dist = self.bucket_mean(ap_dist, mask, 2)
        intra_class_sq_diff = dist[0]
        inter_class_sq_diff = dist[1]
        return intra_class_sq_diff, inter_class_sq_diff

    def loss_fn_training_op(self, x, y, z, logits, class_means):
        """ Computes the loss functions and creates the update ops.

        :param x - input X
        :param y - labels y
        :param z - z layer transform of X.
        :param logits - softmax logits if ce loss is used. Can be None if only ii-loss.
        :class_means - the class means.
        """

        self.intra_c_loss, self.inter_c_loss = self.inter_separation_intra_spread(z, torch.tensor(y, dtype=torch.int),
                                                                                  class_means)

        if self.enable_intra_loss and self.enable_inter_loss:  # The correct ii-loss
            self.loss = torch.mean(self.intra_c_loss - self.inter_c_loss)

        else:
            self.loss = torch.mean(((self.intra_c_loss * 1. if self.enable_intra_loss else 0.)
                                    - (self.inter_c_loss * 1. if self.enable_inter_loss else 0.)
                                    ))

        # Classifier loss
        if self.enable_ce_loss:
            self.ce_loss = torch.mean(nn.functional.cross_entropy(logits, y))

        tvars = self.model.parameters()  # Assuming 'model' is your model

        # Separate trainable variables into different groups
        e_vars = [var for var in tvars if 'enc_' in var.name]
        classifier_vars = [var for var in tvars if 'enc_' in var.name or 'classifier_' in var.name]

        # if self.enable_inter_loss or self.enable_intra_loss:
        #     train_op = self.opt.minimize(self.loss, var_list=e_vars)
        #
        # if self.enable_ce_loss:
        #     ce_train_op = self.c_opt.minimize(self.ce_loss, var_list=classifier_vars)

    def update_class_stats(self, X, y):
        """Recalculates class means and, optionally, covariances in PyTorch."""
        self.model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            z = self.latent(X)  # Assumes latent() returns a PyTorch tensor

        # Calculate class means
        unique_classes = torch.unique(y)
        class_means = {}
        for c in unique_classes:
            class_indices = (y == c)
            class_means[c.item()] = z[class_indices].mean(dim=0)

        self.c_means = class_means

        # Calculate class covariances if necessary
        if self.decision_dist_fn == 'mahalanobis':
            self.c_cov, self.c_cov_inv = self.class_covariance(z, y)

        self.model.train()  # Set the model back to training mode

    def class_covariance(self, Z, y):
        dim = self.z_dim

        per_class_cov = np.zeros((y.shape[1], dim, dim))
        per_class_cov_inv = np.zeros_like(per_class_cov)
        for c in range(y.shape[1]):
            per_class_cov[c, :, :] = np.cov((Z[y[:, c].astype(bool)]).T)
            per_class_cov_inv[c, :, :] = np.linalg.pinv(per_class_cov[c, :, :])

        return per_class_cov, per_class_cov_inv

    def latent(self, X, reformat=True):
        """Computes the z-layer output in PyTorch."""
        self.model.eval()  # Set the model to evaluation mode
        z = torch.zeros((X.shape[0], self.z_dim), device=self.device)  # Ensure tensor is on the correct device

        # Assume batch_size is defined and accessible
        batch_size = self.batch_size

        # Process each batch
        for i in range(0, X.shape[0], batch_size):
            start = i
            end = min(i + batch_size, X.shape[0])

            # Extract the batch and potentially reformat it
            batch_X = X[start:end]
            if reformat:
                batch_X = self.x_reformat(batch_X)  # Ensure x_reformat is adapted for PyTorch

            # Move batch to the appropriate device
            batch_X = batch_X.to(self.device)

            # Perform the forward pass to get the latent representation
            # Assuming self.z_test is replaced with the appropriate PyTorch model forward pass
            z[start:end] = self.model(batch_X).detach()  # Ensure gradients are not calculated

        self.model.train()  # Set the model back to training mode
        return z.cpu().numpy()  # Convert back to numpy array if necessary

    def distance_from_all_classes(self, X, reformat=True):
        """ Computes the distance of each instance from all class means.
        """
        z = self.latent(X, reformat=reformat)
        dist = np.zeros((z.shape[0], self.y_dim))
        for j in range(self.y_dim):
            if self.decision_dist_fn == 'euclidean':  # squared euclidean
                dist[:, j] = np.sum(np.square(z - self.c_means[j]), axis=1)
            elif self.decision_dist_fn == 'mahalanobis':
                dist[:, j] = scipy.spatial.distance.cdist(
                    z, self.c_means[j][None, :],
                    'mahalanobis', VI=self.c_cov_inv[j]).reshape((z.shape[0]))
            else:
                ValueError('Error: Unsupported decision_dist_fn "{0}"'.format(self.decision_dist_fn))

        return dist

    def decision_function(self, X):
        """ Computes the outlier score. The larger the score the more likely it is an outlier.
        """
        dist = self.distance_from_all_classes(X)
        return np.amin(dist, axis=1)

    def predict_prob(self, val_loader, reformat=True):
        """Predicts class probabilities for X over known classes in PyTorch."""
        self.model.eval()  # Ensure the model is in evaluation mode
        probs = []  # List to store all probabilities

        if self.enable_ce_loss:
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    # Move batch to the appropriate device
                    batch_X = batch_X.to(self.device)

                    # Perform the forward pass to get the prediction probabilities
                    logits = self.model(batch_X)
                    prob = nn.functional.softmax(logits, dim=1)
                    probs.append(prob)  # Store probabilities

        elif self.enable_inter_loss and self.enable_intra_loss:
            with torch.no_grad():

                for batch_X, _ in val_loader:
                    # Assume `self.distance_from_all_classes` is adapted for DataLoader input and PyTorch
                    dist = self.distance_from_all_classes(batch_X.to(self.device), reformat=reformat)  # This method needs adaptation

                    prob = torch.exp(-dist)
                    prob = prob / prob.sum(axis=1, keepdim=True)
                    probs.append(prob)  # Store probabilities

        probs = torch.cat(probs)

        self.model.train()  # Set the model back to training mode

        return probs








