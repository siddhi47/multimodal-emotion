from django.shortcuts import render
from django.http import HttpResponse
from webapp.serializers import MultimodalSerializer
from webapp.models import Multimodal
from rest_framework import  generics

def make_prediction(request):
    return HttpResponse("Hi")


class ListViewModels(generics.ListCreateAPIView):
    queryset = Multimodal.objects.all()
    serializer_class = MultimodalSerializer

    def perform_create(self,serializer):
        serializer.save()


