import os
import tempfile
import uuid
import numpy as np
from collections import Counter
from django.http import FileResponse
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .dgcnn_inference import DGCNNInference
from django.shortcuts import render

@api_view(['POST'])
def classify_pointcloud(request):
    file = request.FILES.get('file')
    if not file:
        print('Error: No file provided')
        return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

    print('Received file:', file.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
        for chunk in file.chunks():
            tmp_file.write(chunk)
        tmp_file_path = tmp_file.name
        print('Temporary file created:', tmp_file_path)

    try:
        inferencer = DGCNNInference()
        print('Running DGCNNInference.predict...')
        classified_data = inferencer.predict(tmp_file_path)
        print('Classified data shape:', classified_data.shape)

        # Charger le fichier pour récupérer les 5 premières lignes (for preview)
        print('Loading .npy file...')
        data = np.load(tmp_file_path)
        print('Data shape:', data.shape)
        sample_data = data[:5, :-1]  # Exclure la colonne de classification for preview
        print('Sample data shape:', sample_data.shape)

        # Calcul des stats sur les prédictions (dernière colonne)
        predictions = classified_data[:, -1]  # Prendre seulement la colonne des classes
        print('Predictions shape:', predictions.shape)
        total_points = len(predictions)
        print('Total points:', total_points)
        stats = dict(Counter(predictions))
        print('Stats:', stats)

        # Include the full point cloud (original data + classifications)
        full_point_cloud = classified_data  # Includes x, y, z, ReturnNumber, NumberOfReturns, class

        # Save classified data for download
        download_id = str(uuid.uuid4())
        download_path = os.path.join('media', f'classified_{download_id}.npy')
        os.makedirs('media', exist_ok=True)
        np.save(download_path, classified_data)

        # Construct response
        response_data = {
            'status': 'success',
            'shape': list(data[:, :-1].shape),
            'num_points': int(total_points),
            'stats': {str(k): int(v) for k, v in stats.items()},
            'predictions': predictions.tolist(),
            'sampleData': sample_data.tolist(),
            'fullPointCloud': full_point_cloud.tolist(),
            'downloadUrl': f'/pointscloud_upload/api/download/{download_id}/'
        }
        print('Backend Response:', response_data)

        os.remove(tmp_file_path)
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        print('Error occurred:', str(e))
        os.remove(tmp_file_path)
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def download_classified_file(request, download_id):
    file_path = os.path.join('media', f'classified_{download_id}.npy')
    if not os.path.exists(file_path):
        return Response({'error': 'File not found'}, status=status.HTTP_404_NOT_FOUND)

    try:
        response = FileResponse(open(file_path, 'rb'), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename="classified_{download_id}.npy"'
        # Optionally delete the file after download
        os.remove(file_path)
        return response
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
def upload_page(request):
    return render(request, 'pointscloud_upload/pointscloud_upload.html')