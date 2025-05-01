"""
DGCNN (Dynamic Graph CNN) implementation for point cloud classification.
Original paper: https://arxiv.org/abs/1801.07829
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNN(nn.Module):
    """Dynamic Graph CNN for point cloud classification."""

    def __init__(self, num_classes, k=20):
        """
        Args:
            num_classes: Number of output classes
            k: Number of nearest neighbors for graph construction
        """
        super(DGCNN, self).__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(5 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2))

        # Global feature layer
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2))

        # Classification layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """Forward pass."""
        batch_size = x.size(0)

        # Add artificial point dimension
        x = x.unsqueeze(1).permute(0, 2, 1)

        # EdgeConv blocks
        x1 = self.get_graph_feature(x)
        x1 = self.conv1(x1).max(dim=-1, keepdim=False)[0]

        x2 = self.get_graph_feature(x1)
        x2 = self.conv2(x2).max(dim=-1, keepdim=False)[0]

        x3 = self.get_graph_feature(x2)
        x3 = self.conv3(x3).max(dim=-1, keepdim=False)[0]

        x4 = self.get_graph_feature(x3)
        x4 = self.conv4(x4).max(dim=-1, keepdim=False)[0]

        # Global feature
        x5 = self.conv5(x4).max(dim=-1, keepdim=False)[0]

        # Classification
        x = F.leaky_relu(self.fc1(x5), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def get_graph_feature(self, x):
        """Construct dynamic graph feature."""
        batch_size, num_dims, num_points = x.size()

        # Handle single-point case
        if num_points == 1:
            x = x.repeat(1, 1, self.k)
            num_points = self.k

        # Calculate pairwise distances
        x_t = x.permute(0, 2, 1)
        inner = -2 * torch.matmul(x_t, x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.permute(0, 2, 1)

        # Get k nearest neighbors
        k = min(self.k, num_points - 1)
        idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][..., 1:]

        # Gather neighbors
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)

        x = x.permute(0, 2, 1).contiguous()
        neighbors = x.view(batch_size * num_points, -1)[idx, :]
        neighbors = neighbors.view(batch_size, num_points, k, num_dims)

        # Center and concat features
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((neighbors - x, x), dim=3).permute(0, 3, 1, 2)

        return feature