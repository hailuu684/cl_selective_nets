import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 10 classes in STL-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MySimpleCNN(nn.Module):
    def __init__(self, num_classes=10, num_tasks=5):
        super(MySimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Main task layers
        self.fc1_main = nn.Linear(16 * 5 * 5, 120)
        self.fc2_main = nn.Linear(120, 84)
        self.fc3_main = nn.Linear(84, num_classes)

        # Task inference layers
        self.fc1_task = nn.Linear(16 * 5 * 5, 120)
        self.fc2_task = nn.Linear(120, 1)

    def forward(self, x, return_task_id=False):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        # Main task branch
        x_main = nn.functional.relu(self.fc1_main(x))
        x_main = nn.functional.relu(self.fc2_main(x_main))
        x_main = self.fc3_main(x_main)

        if return_task_id:
            # Task inference branch
            x_task = nn.functional.relu(self.fc1_task(x))
            x_task = torch.sigmoid(self.fc2_task(x_task))  # Applying sigmoid to get a probability
            return x_main, x_task

        else:
            return x_main


class SimpleCNN_WithSelectiveSubnets(nn.Module):
    def __init__(self, num_classes=10, num_child_models=5):
        super().__init__()

        self.subnets = nn.ModuleList([MySimpleCNN(num_tasks=num_child_models, num_classes=num_classes)
                                      for _ in range(num_child_models)])

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image, labels=None, task=None, mode='train'):

        if task is not None and mode == 'train':
            logits, predicted_taskid = self.subnets[task](image, return_task_id=True)

        elif task is None and mode == 'eval_agnostic':

            logits = None

            for net in self.subnets:
                subnet_logits, is_this_task_id = net(image, return_task_id=True)

                if is_this_task_id == 1:
                    logits = subnet_logits

                    #todo: select based on minimum loss


        else:
            raise Exception("Choose the correct mode: train or eval")

        return logits

