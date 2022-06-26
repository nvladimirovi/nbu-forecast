from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

import api.views as views

urlpatterns = [
    path('predict/', views.predict_api),
]

urlpatterns = format_suffix_patterns(urlpatterns)
