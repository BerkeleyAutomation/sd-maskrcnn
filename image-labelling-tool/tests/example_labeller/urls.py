from django.conf.urls import url

from . import views

urlpatterns = [
    # Examples:
    # url(r'^$', 'labelling_aas.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^$', views.home, name='home'),
    url(r'^labelling_tool_api', views.LabellingToolAPI.as_view(), name='labelling_tool_api'),
]
