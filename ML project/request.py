import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':50,'sex':1,'cp':2,'trestbps':120,'chol':126,'fbs':200,'restecg':1,'thalach':88,'slope':1,'thal':2})

print(r.json())