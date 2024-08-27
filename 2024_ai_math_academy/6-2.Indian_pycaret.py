"""
PyCaret으로 인디언 당뇨병 예측하기
"""

import pandas as pd
from pycaret.classification import * 


dataset = pd.read_csv('pima-indians-diabetes.csv',
                      names=['임신횟수', '포도당농도', '혈압', '피부주름두께', '인슐린', '체질량', '혈통', '나이', '결과'])
dia_clf = setup(data=dataset,
                target='결과',
                numeric_features=['임신횟수'],
                train_size=0.8,
                normalize=True,
                session_id=123)
compare_models(sort='AUC')
models()