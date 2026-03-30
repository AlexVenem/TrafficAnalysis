# TrafficAnalysis

Для виртуального окружения нужны:
numpy pandas scikit-learn pytest

# Pipeline parsing

Использование пайплайна:
    python app.py <path/to/hh.csv>
    python app.py <path/to/hh.csv> --output-dir /tmp/out
    python app.py <path/to/hh.csv> -v          # подробные логи

# salary_regression

Подпроект для репозитория `TrafficAnalysis`, который обучает регрессионную модель
на `x_data.npy` / `y_data.npy`, сохраняет веса в `resources/`
и умеет предсказывать зарплаты из выхода существующего пайплайна.

python -m salary_regression.train </path/to/x_data.npy> </path/to/y_data.npy>
python .\salary_regression\app.py x_data.npy