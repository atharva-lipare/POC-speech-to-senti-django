from django.shortcuts import render
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from analysis import analysisAPI
import json
from demosite.settings import STATICFILES_DIRS


# Create your views here.

def home(request):
    return render(request, 'analysis/home.html')


def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'url.json'), 'w') as json_file:
            json.dump({'url': context['url']}, json_file)
        analysisAPI.start_analysis(fs.get_valid_name(name))
        file = open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json'))
        result = file.read()
        file.close()
        with open(os.path.join(STATICFILES_DIRS[0], 'JSON', 'result.json')) as f:
            res1 = json.load(f)
        return render(request, 'analysis/upload.html',
                      {'result': result, 'url': context['url'], 'alltext': str(res1['allText']),
                       'finTones': res1['finTones']})
    return render(request, 'analysis/upload.html')
