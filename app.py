import streamlit as st
from defs import *

st.subheader ('Моделирование продаж')

#data = st.file_uploader('Загрузите данные')
#if data is not None:
#    data = data.getvalue()
#    st.write(data)
#   if st.button('Показать данные'):
#        st.table(data.head())
#else:
data=pd.read_excel('data.xlsx')
data = data_preprocess(data)
if st.button('Показать данные'):
    st.table(data.head())

feature_cols = st.multiselect(
    'Выберите фичи', data.columns.to_list())
target = st.multiselect(
    'Выберите таргет', data.columns.to_list())


#Задаем параметры
N_PAST = int(st.number_input('Сколько недель прошлого берем'))
if N_PAST is None:
    N_PAST = 4

N_FUTURE = int(st.number_input('Горизонт предикта'))
if N_FUTURE is None:
    N_FUTURE = 29


#NeuralNetworkParams

input_size = N_PAST
output_size = N_FUTURE
hidden_size=256
model = LSTM_for_features(input_size, hidden_size, output_size, n_layers=2, n_hidden=256)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.1, gamma=0.1)
num_epochs = 500

#target='KPI \nданные понедельно Продажи, рубли'
#feature_cols = data.drop(columns=['KPI \nданные понедельно Продажи, рубли', 'KPI \nданные понедельно Продажи, упаковки']).columns
#feature_cols = data.drop(columns=['KPI \nданные понедельно Продажи, рубли']).columns
batch_size = 16
subm = pd.read_csv('sample_submission.csv')
pred_data=subm
scaler=StandardScaler()

train_loader, test_loader, val_loader = dataloaders(data, target, feature_cols, N_PAST, N_FUTURE, batch_size)

pred_data = LSTM_go(feature_cols,
            batch_size,
            model,
            train_loader,
            val_loader,
            num_epochs,
            optimizer,
            criterion,
            scheduler,
            pred_data)

preds, model = Catboost_go(data, feature_cols, target)

importance = model.feature_importances_
names = range(len(feature_cols))
model_type = 'Catboost'

#st.write(plot_feature_importance(importance,names,model_type))

feature_table = feature_importance_table(importance, names, feature_cols)
st.table(feature_table)

predicted = predict_cash(pred_data, model, subm)
st.download('Получить предикт')
#st.dataframe(predicted)