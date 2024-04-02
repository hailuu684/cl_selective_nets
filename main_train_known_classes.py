import torch
from models import EncoderDecoder, VAE, AE
from get_datasets import cifar10
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from math import exp
from train_epoch import train_vae_model, train_ae_model


class OptimumThresholdFinder:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

    def predict_known_unknown(self, images, labels, threshold):
        self.model.eval()
        with torch.no_grad():
            logits, reconstructed = self.model(images)
            mse_loss = F.mse_loss(reconstructed, images, reduction='none')
            mse_errors = mse_loss.mean([1, 2, 3])  # Mean over C, H, W
            _, predicted_classes = torch.max(logits, 1)

        predictions = []
        for i, mse in enumerate(mse_errors):
            if mse.item() > threshold:
                predictions.append(-1)  # Mark as unknown
            else:
                predictions.append(predicted_classes[i].item())

        return predictions

    def compute_metric(self, predictions, true_labels):
        return accuracy_score(true_labels, predictions)

    def find_optimum_threshold(self, val_loader, thresholds):
        best_threshold = None
        best_metric = -float('inf')

        for threshold in thresholds:
            all_predictions = []
            all_true_labels = []

            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).tolist()
                predictions = self.predict_known_unknown(images, labels, threshold)

                all_predictions.extend(predictions)
                all_true_labels.extend(labels)

            # Adjust metric calculation as necessary
            metric = self.compute_metric(all_predictions, all_true_labels)

            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold

        return best_threshold, best_metric


def train(model, train_loader, val_loader, optimizer, num_epochs,
          classification_loss_fn, reconstruction_loss_fn, known_classes, device,
          threshold=0.1):
    thresholds = torch.arange(0.01, 0.5, 0.01)
    alpha = 1.5

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits, reconstructed = model(images)

            # Calculate losses
            classification_loss = classification_loss_fn(logits, labels)
            reconstruction_loss = reconstruction_loss_fn(reconstructed, images)

            # Combine losses
            loss = classification_loss + alpha * reconstruction_loss
            running_loss += loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Acc
            _, predicted = torch.max(logits.data, 1)
            correct_predictions += (predicted == labels).sum().item()

            total_predictions += labels.size(0)

        epoch_loss = running_loss / total_predictions
        epoch_accuracy = correct_predictions / total_predictions

        # todo: implement optimum threshold finder
        # # Find optimum threshold after each epoch or after specific epochs
        # if epoch % 1 == 0:  # Example: after every epoch
        #     evaluator = OptimumThresholdFinder(model, device)
        #     best_threshold, _ = evaluator.find_optimum_threshold(val_loader, thresholds)
        # else:
        #     best_threshold = threshold  # Use previous best or default

        val_loss, val_accuracy, val_unknown_acc = validate_model(model, val_loader,
                                                                 known_classes, classification_loss_fn,
                                                                 threshold=0.1,
                                                                 device=device)

        print(
            f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, Val Unknown Accuracy: {val_unknown_acc:.2f}")


def validate_model(model, val_loader, known_classes, classification_loss_fn, device, threshold=0.1):
    model.eval()  # Set the model to evaluation mode

    # Correct closed prediction
    total_loss = 0.0
    correct_predictions = 0
    total = 0

    # open prediction
    found_unknown = 0  # Poorly reconstructed or misclassified as unknown
    actual_unknown = 0  # Truly unknown (labels outside known_classes)
    total_unknown = 0

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            logits, reconstructed = model(images)

            # Get metrics
            val_loss = classification_loss_fn(logits, labels)

            total_loss += val_loss.item() * images.size(0)

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            # Compare reconstructed images with original images
            mse_loss = torch.nn.MSELoss(reduction='none')
            mse_errors = mse_loss(reconstructed, images).mean([1, 2, 3])  # Average over all dimensions except the batch

            # Deciding if known or unknown based on threshold
            for i, mse in enumerate(mse_errors):
                if mse >= threshold and predicted[i] != labels[i]:
                    found_unknown += 1
                if labels[i].item() not in known_classes:
                    actual_unknown += 1
                elif predicted[i] == labels[i]:
                    correct_predictions += 1

            # Get total unknown in test set
            total_unknown += torch.sum(labels > len(known_classes))

    avg_loss = total_loss / total
    known_accuracy = correct_predictions / total
    unknown_accuracy = found_unknown / (actual_unknown if actual_unknown > 0 else 1)  # Avoid division by zero

    return avg_loss, known_accuracy, unknown_accuracy


def main():
    known_classes = [0, 1, 2, 3, 4]
    train_loader, val_loader, test_all_cls_loader = cifar10.get_known_class_data(batch_size=64,
                                                                                 known_classes=known_classes)

    num_classes = 10  # CIFAR-10 has 10 classes

    model = EncoderDecoder.HybridResNet(block=EncoderDecoder.ResidualBlock, layers=[2, 2, 2], num_classes=num_classes)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)

    # Loss functions
    classification_loss_fn = nn.CrossEntropyLoss()
    reconstruction_loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model=model, train_loader=train_loader, val_loader=test_all_cls_loader, optimizer=optimizer,
          num_epochs=9, classification_loss_fn=classification_loss_fn, reconstruction_loss_fn=reconstruction_loss_fn,
          known_classes=known_classes, device=device)

    # Save model
    torch.save(model, '/home/luu/projects/cl_selective_nets/results/train_known_hybrid_model.pt')


def main_vae():
    known_classes = [0, 1, 2, 3, 4]
    train_loader, val_loader, test_all_cls_loader = cifar10.get_known_class_data(batch_size=64,
                                                                                 known_classes=known_classes)

    num_classes = 10  # CIFAR-10 has 10 classes

    model = VAE.VAEWithClassifier(image_size=32, channel_num=3, kernel_num=128, z_size=128, num_classes=num_classes)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)

    train_vae_model(model, train_loader, test_all_cls_loader, known_classes=known_classes, epochs=100, device=device)

    # Save model
    torch.save(model.state_dict(), '/home/luu/projects/cl_selective_nets/results/vae_model_state_dict.pt')


def main_ae():
    known_classes = [0, 1, 2, 3, 4]
    train_loader, val_loader, test_all_cls_loader = cifar10.get_known_class_data(batch_size=64,
                                                                                 known_classes=known_classes)

    num_classes = 10  # CIFAR-10 has 10 classes

    latent_dim = 1024
    model = AE.AEWithClassifier(base_channel_size=32, latent_dim=latent_dim, num_input_channels=3,
                                width=32, height=32, num_classes=num_classes)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)

    train_ae_model(model, train_loader, test_all_cls_loader, known_classes=known_classes, epochs=100, device=device)

    # Save model
    torch.save(model.state_dict(), f'/home/luu/projects/cl_selective_nets'
                                   f'/results/ae_model_state_dict_{latent_dim}-latent_add-maxpooling.pt')


# ---------------------------------------------------------------------------------
# -------------------------- EVALUATION -----------------------------------------
# ---------------------------------------------------------------------------------

def gaussian_window(size, sigma):
    gauss = torch.Tensor([exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def load_saved_model(evaluate_mode='mse'):
    model = torch.load('/home/luu/projects/cl_selective_nets/results/train_known_hybrid_model.pt')

    model.eval()

    known_classes = [0, 1, 2, 3, 4]
    threshold = 0.15
    train_loader, val_loader, test_all_cls_loader = cifar10.get_known_class_data(batch_size=64,
                                                                                 known_classes=known_classes)

    test_loader = test_all_cls_loader

    image_indices = torch.arange(0, len(test_loader.dataset))

    total_unknown = 0
    total_known = 0
    wrong_unknown_prediction = 0
    correct_unknown_prediction = 0
    correct_known_prediction = 0
    wrong_known_prediction = 0

    for batch, image_index in enumerate(image_indices):

        # Directly access the dataset
        if image_index < len(test_loader.dataset):
            images, labels = test_loader.dataset[image_index]
        else:
            print("Image index is out of range.")
            return

        # Add a batch dimension if necessary
        images = images.unsqueeze(0)
        labels = torch.tensor([labels])  # Ensure labels is a tensor

        # Move to the same device as the model
        images = images.to(next(model.parameters()).device)
        labels = labels.to(next(model.parameters()).device)

        with torch.no_grad():
            logits, reconstructed = model(images)

            if evaluate_mode == 'mse':
                mse_loss = torch.nn.MSELoss(reduction='none')
                mse_errors = mse_loss(reconstructed, images).mean([1, 2, 3])  # Mean over channels, height, and width
                mse_error = mse_errors[0].item()

            elif evaluate_mode == 'ssim':
                ssim_value = ssim(reconstructed, images)

            _, predicted = torch.max(logits, 1)

        # # Prepare the images for visualization
        # real_img = images[0].cpu().numpy().transpose((1, 2, 0))  # Convert to HWC format
        # recon_img = reconstructed[0].cpu().numpy().transpose((1, 2, 0))  # Convert reconstructed image to HWC format

        true_label = labels[0].item()
        pred_label = predicted[0].item()

        if true_label not in known_classes:
            total_unknown += 1
        else:
            total_known += 1

        if evaluate_mode == 'mse':
            # if known_classes is not None and true_label not in known_classes:
            #     annotation = f"Predicted: Unknown\nMSE = {mse_error}"
            if mse_error >= threshold:
                annotation = f"Predicted: Unknown due to high MSE = {mse_error}\n" \
                             f"Predicted: {pred_label} - True: {true_label}"

                if pred_label == true_label:
                    wrong_unknown_prediction += 1
                else:
                    correct_unknown_prediction += 1
            else:
                annotation = f"Predicted: {pred_label} - True: {true_label}\nMSE = {mse_error}"

                if pred_label == true_label:
                    correct_known_prediction += 1
                else:
                    wrong_known_prediction += 1

        elif evaluate_mode == 'ssim':
            if ssim_value > 0.7:
                annotation = f'Known class - SSIM: {ssim_value}\n' \
                             f'Predicted: {pred_label} - True: {true_label}'
            else:
                annotation = f'Unknown class - SSIM: {ssim_value}\n' \
                             f'Predicted: {pred_label} - True: {true_label}'

        else:
            raise Exception("Choose correct evaluation mode: MSE or SSIM")

        if batch % 100 == 0:
            print(annotation)
            print(" ")

    print(f'correct_unknown_prediction: {correct_unknown_prediction / total_unknown}\n'
          f'wrong_unknown_prediction: {wrong_unknown_prediction / total_unknown}\n'
          f'correct_known_prediction: {correct_known_prediction / total_known}\n'
          f'wrong_known_prediction: {wrong_known_prediction / total_known}')


if __name__ == '__main__':
    main_ae()
    # main_vae()
    # main()
    # load_saved_model(evaluate_mode='ssim')
