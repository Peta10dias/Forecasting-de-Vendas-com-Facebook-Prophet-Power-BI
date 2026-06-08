import pandas as pd
from prophet import Prophet

# Carregar os dados 
orders = pd.read_csv('olist_orders_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')

df_merged = pd.merge(orders, items, on='order_id', how='inner')
df_merged = pd.merge(df_merged, products, on='product_id', how='left')
df_merged['order_purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp'])

# Identificar as 10 Categorias
top_10_categories = df_merged['product_category_name'].value_counts().head(10).index.tolist()


all_forecasts = []


# Loop de Automação
for categoria in top_10_categories:
    print(f"Processando: {categoria}...")
    
    # Filtro
    df_cat = df_merged[df_merged['product_category_name'] == categoria].copy()
    df_day = df_cat.groupby(df_cat['order_purchase_timestamp'].dt.date).agg({'price': 'sum'}).reset_index()
    df_day.columns = ['ds', 'y']
    df_day['ds'] = pd.to_datetime(df_day['ds']) # Evita o erro de merge no Power BI
    
    #Treinar Modelo
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_country_holidays(country_name='BR')
    model.fit(df_day)
    
    # Gerar Previsão (90 dias)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    
    forecast_final = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
        df_day[['ds', 'y']], on='ds', how='left'
    )
    forecast_final.rename(columns={'y': 'realizado'}, inplace=True)
    forecast_final['product_category_name'] = categoria
    
    # Guardar o resultado
    all_forecasts.append(forecast_final)

df_final_consolidado = pd.concat(all_forecasts, ignore_index=True)
df_final_consolidado.to_csv('olist_previsoes_top10.csv', index=False)