from django.shortcuts import render

import os
import pickle
import numpy as np

modulePath = os.path.dirname(__file__)
filePath = os.path.join(modulePath, 'finalized_model.model')
with open(filePath, 'rb') as f:
    model = pickle.load(f)

def home(request):
    return render(request, 'index.html')

def classify(request):
    n = float(request.POST.get('nitrogen'))
    p = float(request.POST.get('phosphorus'))
    k = float(request.POST.get('potassium'))
    temp = float(request.POST.get('temperature'))
    humidity = float(request.POST.get('humidity'))
    ph = float(request.POST.get('ph'))
    rainfall = float(request.POST.get('rainfall'))

    features = [n, p, k, temp, humidity, ph, rainfall]
    features = np.array(features)
    features = features.reshape(1, -1)
    predicted = model.predict(features)[0]

    context = {'crop': predicted}
    return render(request, 'index.html', context)
    