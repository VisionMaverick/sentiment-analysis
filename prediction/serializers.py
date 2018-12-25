from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    prediction = serializers.CharField()
    score = serializers.FloatField()
