
from rest_framework import serializers
from webapp.models import Multimodal, Predictions

class MultimodalSerializer(serializers.ModelSerializer):
    class Meta:
        model = Multimodal
        fields = '__all__'


