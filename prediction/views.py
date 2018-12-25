import pickle

import nltk
import pandas as pd
from rest_framework import views
from rest_framework.response import Response
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from .serializers import PredictionSerializer


# ----------------------------------
# Prediction View for REST
# ---------------------------------

class Prediction(views.APIView):

    @staticmethod
    def read_dataset(file_url):
        df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
        return df

    def get(self, request):
        data = request.query_params.get('data', None)
        data = data.lower()
        data = data.split()

        bigram_finder = BigramCollocationFinder.from_words(data)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 200)
        d = dict([(bigram, True) for bigram in bigrams])
        #d = dict([(word, True) for word in data])
        estimator = pickle.load(open('classifier.pkl', 'rb'))
        print(d)
        result = estimator.classify(d)

        predictions = []
        predictions.append({
            'prediction': result,
            'score': 1.0
        })

        response = PredictionSerializer(predictions, many=True).data
        print("........................response.......{}".format(response))
        return Response(response)
