# RNN Gold Filter — советник для XAUUSD (MetaTrader 5)

Советник на MQL5, который фильтрует торговые сигналы по золоту двумя GRU-нейросетями
(отдельная сеть для BUY и для SELL), экспортированными в ONNX. Обучение выполняется в Python (PyTorch).

## Как это работает

```
PrepareData.mq5 ──► data/*.csv ──► python/feature_pipeline.py ──► artifacts/*.npz
                                                                     │
                                    notebooks/rnn_learn.ipynb  (подбор гиперпараметров, Optuna)
                                    notebooks/rnn_final_train.ipynb / python/run_final_train.py
                                                                     ▼
                                   output/buy_model.onnx, sell_model.onnx, RNN_Scaler.mqh
                                                                     ▼
                                                         RNNFilter_Gold.mq5 (советник)
```

1. **Выгрузка данных** — скрипт `PrepareData.mq5` сохраняет окна из 20 баров × 18 сырых признаков
   (OHLCV, EMA 8/21, RSI, Stochastic, MACD, ATR, время) и метку результата сделки.
2. **Предобработка** — `python/feature_pipeline.py` превращает их в тензор 20×23
   (признаки нормируются на ATR, z-score по строке). Контракт признаков: `artifacts/model_features.json`.
3. **Обучение** — GRU-сети (PyTorch), подбор параметров через Optuna, экспорт в ONNX.
4. **Проверка** — `python/test_net.py` прогоняет ONNX-модели по сырым CSV через тот же пайплайн.
5. **Советник** — `RNNFilter_Gold.mq5` грузит модели и торгует.

## Логика советника

- Одновременно открыта только одна сделка, фиксированный лот.
- BUY разрешён, если BUY-сеть уверена в покупке (confidence ≥ порога) **и** SELL-сеть не уверена в продаже; SELL — зеркально.
- SL и TP (RR) в пунктах, опционально трейлинг, безубыток, новостной фильтр.
- Настройки для тестера: `Set/word_RNN_gold.set`.

## Структура проекта

| Путь | Назначение |
|---|---|
| `RNNFilter_Gold.mq5` | Советник |
| `RNN_ONNX.mqh` | Загрузка ONNX и инференс |
| `RNN_Scaler.mqh` | Параметры нормализации (генерируются при обучении) |
| `RNNFilter_News.mqh` | Новостной фильтр |
| `TargetWinRateCriterion.mqh` | Критерий оптимизации по win rate |
| `PrepareData.mq5` | Выгрузка обучающих данных |
| `Set/` | Наборы параметров для тестера |
| `python/` | Предобработка, финальное обучение, проверка моделей |
| `notebooks/` | Ноутбуки: подбор гиперпараметров и финальное обучение |
| `data/`, `artifacts/`, `output/` | Данные и результаты (не в git, воссоздаются скриптами) |

MQL5-файлы лежат в корне, потому что советник подключает заголовки по относительным путям
и должен находиться в `MQL5/Experts/`.

## Запуск

Все команды выполняются из корня проекта.

```bash
pip install -r requirements.txt
python python/feature_pipeline.py                 # data/*.csv -> artifacts/*.npz
jupyter notebook notebooks/rnn_learn.ipynb        # подбор параметров и обучение
python python/run_final_train.py                  # финальное обучение (результат в output_final/)
python python/test_net.py --onnx-dir output       # проверка ONNX-моделей
```

Затем скопируйте `buy_model.onnx` и `sell_model.onnx` в `MQL5/Files/RNN_GOLD/`,
`RNN_Scaler.mqh` — рядом с советником, и скомпилируйте `RNNFilter_Gold.mq5` в MetaEditor.

## Результаты обучения (`output/summary.json`)

| | BUY | SELL |
|---|---|---|
| Сделок | 488 | 1693 |
| Win rate | 50.6 % | 49.1 % |
| Profit factor | 1.04 | 0.95 |

SL 15 / TP 20 пипсов. Это исследовательский прототип: BUY-модель около нуля, SELL-модель
убыточна на этой выборке, статистически значимого преимущества пока нет.
