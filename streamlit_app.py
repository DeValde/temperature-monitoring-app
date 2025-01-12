# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import aiohttp
import asyncio
from datetime import datetime
from io import StringIO
import nest_asyncio
import warnings
nest_asyncio.apply()
warnings.simplefilter(action='ignore', category=DeprecationWarning)
sns.set(style="darkgrid")



def get_season(month):
    month_to_season = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    }
    return month_to_season.get(month, "unknown")


def get_current_temperature_sync(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric',
        'lang': 'ru'
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 401:
            return None, data.get("message", "Invalid API key.")
        elif response.status_code != 200:
            return None, data.get("message", "Error fetching data.")

        current_temp = data['main']['temp']
        return current_temp, None
    except Exception as e:
        return None, str(e)


async def get_current_temperature_async(session, city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric',
        'lang': 'ru'
    }

    try:
        async with session.get(base_url, params=params) as response:
            data = await response.json()

            if response.status == 401:
                return None, data.get("message", "Invalid API key.")
            elif response.status != 200:
                return None, data.get("message", "Error fetching data.")

            current_temp = data['main']['temp']
            return current_temp, None
    except Exception as e:
        return None, str(e)


async def fetch_all_temperatures_async(cities, api_key):
    temperatures = {}
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(get_current_temperature_async(session, city, api_key)) for city in cities]
        results = await asyncio.gather(*tasks)
        for city, (temp, error) in zip(cities, results):
            if error:
                temperatures[city] = None  # Устанавливаем None, если есть ошибка
            else:
                temperatures[city] = temp
    return temperatures


def run_async(coroutine):
    try:
        return asyncio.run(coroutine)
    except RuntimeError as e:
        if 'asyncio.run() cannot be called from a running event loop' in str(e):
            # Если цикл событий уже запущен, создаем новый цикл в отдельном потоке
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coroutine)
        else:
            raise e


def is_temperature_normal(city, current_temp, df):
    current_date = datetime.now()
    current_season = get_season(current_date.month)

    df_filtered = df[(df['city'] == city) & (df['season'] == current_season)]

    if df_filtered.empty:
        return False, None, None, None

    mean_temp = df_filtered['temperature'].mean()
    std_temp = df_filtered['temperature'].std()

    upper_bound = mean_temp + 2 * std_temp
    lower_bound = mean_temp - 2 * std_temp

    is_normal = lower_bound <= current_temp <= upper_bound
    return is_normal, mean_temp, lower_bound, upper_bound


def main():
    st.title("Мониторинг Температурных Данных")
    st.write("""
HT1 Panteleev V
    """)

    st.header("1. Загрузка Исторических Данных")
    uploaded_file = st.file_uploader("Загрузите CSV-файл с историческими данными о температуре", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

            required_columns = {'timestamp', 'city', 'temperature', 'season'}
            if not required_columns.issubset(df.columns):
                st.error(f"Загруженный файл должен содержать следующие столбцы: {', '.join(required_columns)}")
                return

            df['season'] = df['season'].str.lower().str.strip()

            duplicates = df.duplicated(subset=['city', 'timestamp'])
            if duplicates.any():
                st.warning(f"Найдены дубликаты: {duplicates.sum()}. Удаление дубликатов.")
                df = df.drop_duplicates(subset=['city', 'timestamp'])
            else:
                st.success("Дубликаты не найдены.")

            df = df.sort_values(['city', 'timestamp'])

            df.set_index('timestamp', inplace=True)
            df['temperature'] = df.groupby('city')['temperature'].transform(lambda group: group.interpolate(method='time'))
            df.reset_index(inplace=True)

            missing = df['temperature'].isnull().sum()
            if missing > 0:
                st.error(f"В данных остались {missing} пропущенных значений после интерполяции.")
            else:
                st.success("Интерполяция завершена успешно.")

            st.subheader("Пример Загруженных Данных")
            st.write(df.head())

            cities = df['city'].unique()
            st.header("2. Настройка Мониторинга")
            city = st.selectbox("Выберите город", options=cities)

            api_key = st.text_input("Введите ваш API-ключ OpenWeatherMap", type="password")
            if city:
                st.header("3. Анализ Исторических Данных")
                st.subheader("Описательная Статистика")
                city_data = df[df['city'] == city]
                desc = city_data['temperature'].describe()
                st.write(desc)
                st.subheader("Временной Ряд Температур с Аномалиями")
                current_date = datetime.now()
                current_season = get_season(current_date.month)
                df_city_season = city_data[city_data['season'] == current_season]

                mean_temp = df_city_season['temperature'].mean()
                std_temp = df_city_season['temperature'].std()

                upper_bound = mean_temp + 2 * std_temp
                lower_bound = mean_temp - 2 * std_temp
                anomalies = city_data[
                    (city_data['temperature'] > upper_bound) | (city_data['temperature'] < lower_bound)]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(city_data['timestamp'], city_data['temperature'], label='Температура')
                ax.axhline(mean_temp, color='green', linestyle='--', label='Среднее')
                ax.axhline(upper_bound, color='red', linestyle='--', label='Верхняя граница')
                ax.axhline(lower_bound, color='red', linestyle='--', label='Нижняя граница')
                ax.scatter(anomalies['timestamp'], anomalies['temperature'], color='orange', label='Аномалии')
                ax.set_xlabel('Дата')
                ax.set_ylabel('Температура (°C)')
                ax.set_title(f'Временной ряд температуры для {city} ({current_season.capitalize()})')
                ax.legend()
                st.pyplot(fig)
                st.subheader("Сезонные Профили")
                season_group = city_data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()

                fig2, ax2 = plt.subplots(figsize=(8, 5))
                colors = sns.color_palette("viridis", len(season_group))
                ax2.bar(season_group['season'], season_group['mean'], yerr=season_group['std'], capsize=5, color=colors, alpha=0.7)

                ax2.set_xlabel('Сезон')
                ax2.set_ylabel('Средняя Температура (°C)')
                ax2.set_title(f'Сезонные профили температуры для {city}')
                ax2.yaxis.grid(True)

                st.pyplot(fig2)
                st.header("4. Текущая Температура и Её Нормальность")

                if api_key:
                    method = st.radio("Выберите метод получения данных:", ("Синхронный", "Асинхронный"))

                    temperatures = {}
                    error_message = None

                    if method == "Синхронный":
                        current_temp, error = get_current_temperature_sync(city, api_key)
                        if error:
                            error_message = error
                        else:
                            temperatures[city] = current_temp
                    else:
                        try:
                            temperatures = run_async(fetch_all_temperatures_async([city], api_key))
                            current_temp = temperatures.get(city, None)
                            if current_temp is None:
                                error_message = "err api key."
                        except Exception as e:
                            error_message = str(e)

                    if error_message:
                        st.error(f"Ошибка при получении текущей температуры: {error_message}")
                    else:
                        current_temp = temperatures.get(city)
                        if current_temp is not None:
                            is_normal, mean, lower, upper = is_temperature_normal(city, current_temp, df)
                            if is_normal:
                                status = "в пределах нормы"
                                color = "green"
                            else:
                                status = "аномалия"
                                color = "red"

                            st.metric(label=f"Текущая температура в {city}", value=f"{current_temp}°C")
                            st.markdown(f"**Статус температуры:** <span style='color:{color}'>{status}</span>",
                                        unsafe_allow_html=True)
                        else:
                            st.error("Не удалось получить текущую температуру.")
                else:
                    st.info("Введите ваш API-ключ OpenWeatherMap для отображения текущей температуры.")

        except Exception as e:
            st.error(f"file err: {e}")
    else:
        st.info("hist file needed.")


# Исправление: Вызов функции main() внутри блока if __name__ == "__main__"
if __name__ == "__main__":
    main()

