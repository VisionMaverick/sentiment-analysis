from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^predictions/$', views.Prediction.as_view()),

]
