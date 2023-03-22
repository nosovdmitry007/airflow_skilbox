# # <YOUR_IMPORTS>
import json
import dill
import pandas as pd
import os
path = os.environ.get('PROJECT_PATH', '..')

mo = os.listdir(f'{path}/data/models/')
m = []
for z in mo:
    k = z.split('_')[-1]
    m.append(k[:-4])

mod = max(m)
# mod = '202303221932'
print(mod)
with open(f'{path}/data/models/cars_pipe_{str(mod)}.pkl', 'rb') as file:
    model = dill.load(file)


def opn(i):
    with open(f'{path}/data/test/{i}') as json_file:
        return json.load(json_file)


def predict():
    # <YOUR_CODE>
    lis = os.listdir(f'{path}/data/test/')
    df_pred = pd.DataFrame()
    file_name = f'{path}/data/predictions/preds_{mod}.csv'

    for i in lis:
        form = opn(i)
        df = pd.DataFrame.from_dict([form])
        y = model.predict(df)
        pred = [{
            'id': form['id'],
            'Result': y[0],
            'price': form['price']
        }]
        dt = pd.DataFrame.from_dict(pred)
        df_pred = pd.concat([df_pred, dt], ignore_index=True)

    df_pred.to_csv(file_name, sep='\t', encoding='utf-8')


if __name__ == '__main__':
    predict()
