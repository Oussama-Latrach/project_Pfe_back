import os
import numpy as np
import torch
import open3d as o3d
from matplotlib import pyplot as plt
from django.conf import settings


class DGCNNInference:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = model_path or os.path.join(settings.BASE_DIR, 'pointscloud_upload', 'models', 'best_model.pth')

        class DGCNN(torch.nn.Module):
            def __init__(self, num_classes=4, k=20):
                super().__init__()
                self.k = k
                self.conv1 = torch.nn.Sequential(
                    torch.nn.Conv2d(5 * 2, 64, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.conv2 = torch.nn.Sequential(
                    torch.nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.conv3 = torch.nn.Sequential(
                    torch.nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.conv4 = torch.nn.Sequential(
                    torch.nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.conv5 = torch.nn.Sequential(
                    torch.nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                    torch.nn.BatchNorm1d(1024),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.fc1 = torch.nn.Linear(1024, 512)
                self.fc2 = torch.nn.Linear(512, 256)
                self.fc3 = torch.nn.Linear(256, num_classes)
                self.dropout = torch.nn.Dropout(p=0.5)

            def forward(self, x):
                batch_size = x.size(0)
                x = x.unsqueeze(1).permute(0, 2, 1)
                x = self._edge_conv_blocks(x)
                x = self._classification(x)
                return x

            def _edge_conv_blocks(self, x):
                x1 = self._get_graph_feature(x)
                x1 = self.conv1(x1).max(dim=-1, keepdim=False)[0]
                x2 = self._get_graph_feature(x1)
                x2 = self.conv2(x2).max(dim=-1, keepdim=False)[0]
                x3 = self._get_graph_feature(x2)
                x3 = self.conv3(x3).max(dim=-1, keepdim=False)[0]
                x4 = self._get_graph_feature(x3)
                x4 = self.conv4(x4).max(dim=-1, keepdim=False)[0]
                x5 = self.conv5(x4).max(dim=-1, keepdim=False)[0]
                return x5

            def _get_graph_feature(self, x):
                batch_size, num_dims, num_points = x.size()
                if num_points == 1:
                    x = x.repeat(1, 1, self.k)
                    num_points = self.k
                x_t = x.permute(0, 2, 1)
                pairwise_distance = -torch.sum(x ** 2, dim=1, keepdim=True) - 2 * torch.matmul(x_t, x) - torch.sum(
                    x ** 2, dim=1, keepdim=True).permute(0, 2, 1)
                k = min(self.k, num_points - 1)
                idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][..., 1:]
                idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
                idx = (idx + idx_base).view(-1)
                x = x.permute(0, 2, 1).contiguous()
                neighbors = x.view(batch_size * num_points, -1)[idx, :]
                neighbors = neighbors.view(batch_size, num_points, k, num_dims)
                x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
                return torch.cat((neighbors - x, x), dim=3).permute(0, 3, 1, 2)

            def _classification(self, x):
                x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.2)
                x = self.dropout(x)
                x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.2)
                x = self.dropout(x)
                return self.fc3(x)

        self.model = DGCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.colors = np.array([
            [1, 0, 0],  # Classe 1 - Rouge
            [0, 1, 0],  # Classe 2 - Vert
            [0, 0, 1],  # Classe 3 - Bleu
            [1, 1, 0],  # Classe 4 - Jaune
        ])

        self.class_names = ['Unclassified', 'Ground', 'Vegetation', 'Building']

    def predict(self, file_path):
        data = np.load(file_path)
        assert data.shape[1] == 5, "Le fichier doit avoir exactement 5 colonnes"
        points = torch.tensor(data, dtype=torch.float32).to(self.device)

        predictions = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                pred = self.model(batch)
                predictions.append(pred.argmax(dim=1).cpu().numpy())

        class_ids = np.concatenate(predictions)
        classified_data = np.column_stack((data, class_ids + 1))
        return classified_data

    def save_results(self, classified_data, original_path):
        results_dir = os.path.join(settings.MEDIA_ROOT, 'classification_results')
        os.makedirs(results_dir, exist_ok=True)

        base_name = os.path.basename(original_path).replace('.npy', '')

        # Sauvegarde des données classifiées
        result_path = os.path.join(results_dir, f'{base_name}_classified.npy')
        np.save(result_path, classified_data)

        # Sauvegarde de la visualisation 2D
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(classified_data[:, 0], classified_data[:, 1],
                              c=self.colors[classified_data[:, 5].astype(int) - 1],
                              s=1)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=self.colors[i],
                                      markersize=8, label=self.class_names[i])
                           for i in range(len(self.class_names))]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title("Classification Results - XY Projection")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')

        img_path = os.path.join(results_dir, f'{base_name}_2d.png')
        plt.savefig(img_path, dpi=300)
        plt.close()

        # Sauvegarde au format PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(classified_data[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(
            self.colors[classified_data[:, 5].astype(int) - 1])

        ply_path = os.path.join(results_dir, f'{base_name}_classified.ply')
        o3d.io.write_point_cloud(ply_path, pcd)

        return {
            'npy_path': result_path,
            'img_path': img_path,
            'ply_path': ply_path,
            'base_name': base_name,
            'stats': self.get_stats(classified_data)
        }

    def get_stats(self, classified_data):
        unique, counts = np.unique(classified_data[:, 5], return_counts=True)
        stats = []
        total = len(classified_data)
        for cls, count in zip(unique, counts):
            stats.append({
                'class_id': int(cls),
                'class_name': self.class_names[int(cls) - 1],
                'count': count,
                'percentage': f"{(count / total) * 100:.2f}%"
            })
        return stats