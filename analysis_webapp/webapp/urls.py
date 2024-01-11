from django.urls import path
from webapp import views
urlpatterns = [

        path("",views.make_prediction),
        path("list_models",views.ListViewModels.as_view())
        ]
