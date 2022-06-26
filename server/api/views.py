from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import torch
from .predictor import Predictor, class_names

predictor = Predictor()

# Create your views here.
@api_view(['GET', 'POST'])
def predict_api(request):
    """
    Categorises fruid in 1 of 33 clasess.

    Example Usage:
    POST
    URL: /predict
    Accepts: binary (image)
    Headers:
        Content-Disposition: attachment; filename=image_name.jpg
    """

    if request.method == 'POST':
        try:
            image = predictor.transform_image(request.data["file"].file)
            outputs = predictor.vgg16(image)

            _, preds = torch.max(outputs.data, 1)
            predicted_labels = [preds[j] for j in range(image.size()[0])]

            return Response(class_names[predicted_labels[0]], status=status.HTTP_200_OK)
        except Exception:
            return Response(status=status.HTTP_400_BAD_REQUEST)
