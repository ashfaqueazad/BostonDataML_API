
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if reg:
        try:
            json_ = request.json
            print(json_)
            
            query = pd.get_dummies(pd.DataFrame(json_))
            
            query = query.reindex(columns = rfholdout_columns_model,fill_value=0)
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            d1 = joblib.load('forStandardScalingRF.pkl')
            d2 = joblib.load('forStandardScalingTestRF.pkl')
            e =StandardScaler()
            d1 = e.fit_transform(d1)
            d2 = e.transform(d2)
            t_query=e.transform(query)
            prediction1 = list(reg.predict(t_query))
            return jsonify({'prediction': str(prediction1)})
        except:
            return jsonify({'trace': traceback.format_exc()})
        

if __name__ == '__main__':

    try:
        port = int(sys.argv[1])
    except:
        port = 12345
    
    reg = joblib.load("rfholdout_model.pkl")
    print ('Model loaded')
    rfholdout_columns_model = joblib.load("rfholdout_columns_model.pkl")
    print ('Model columns loaded')
    
    app.run(port = port,debug=True)    

