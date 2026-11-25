# ИИ-фреймворки — лабораторные работы

Пять лабораторных и итоговое сравнение алгоритмов на двух датасетах с Московской
биржи (MOEX ISS API):

| Задача | Датасет | Что предсказываем |
|---|---|---|
| Классификация | **MOEX Equity Panel** — дневные торги акциями, режим TQBR, топ-50 по обороту | вырастет ли цена закрытия на следующем торговом дне |
| Регрессия | **MOEX Bonds Panel** — дневные торги облигациями, режим TQCB, топ-50 по обороту | доходность к погашению (`YIELDCLOSE`) на следующем торговом дне, % |

Период: 2021-01-01 … 2026-11-25. Разбиение по времени: обучение — до 2025-01-01,
тест — с 2025-01-01 (никакого перемешивания, чтобы не было утечки из будущего).

## Лабораторные

| № | Алгоритм | Ноутбук | Своя реализация |
|---|---|---|---|
| 1 | K ближайших соседей | `notebooks/01_knn.ipynb` | `src/knn.py` |
| 2 | Линейная и логистическая регрессия | `notebooks/02_linear.ipynb` | `src/linear.py` |
| 3 | Дерево решений | `notebooks/03_decision_tree.ipynb` | `src/tree.py` |
| 4 | Случайный лес | `notebooks/04_random_forest.ipynb` | `src/ensemble.py` |
| 5 | Градиентный бустинг | `notebooks/05_gradient_boosting.ipynb` | `src/boosting.py` |
| 6 | Сравнение всех алгоритмов | `notebooks/06_comparison.ipynb` | — |

В каждой лабораторной: бейзлайн на sklearn, своя реализация, проверка гипотез
об улучшении (2–3 варианта параметров/признаков), итоговая таблица метрик
и выводы отдельно для классификации и для регрессии.

## Данные

Сырые выгрузки и признаки в репозиторий не кладутся (объём, условия ISS) —
воспроизводятся двумя командами:

```bash
python scripts/fetch_moex.py        # data/raw/{shares,bonds}.csv + справочники
python scripts/build_features.py    # data/processed/*_features.csv
```

`scripts/russian_trusted_ca.pem` — корневой сертификат Russian Trusted (Минцифры),
им подписана цепочка `iss.moex.com`; он передаётся в `urllib` явно, проверка TLS
не отключается.

## Запуск

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

## Метрики

Классификация: accuracy, precision, recall, F1, ROC-AUC (свои реализации —
`src/metrics.py`). Регрессия: RMSE, MAE, R².

Важно: дневное направление цены предсказуемо плохо, accuracy около 50 % —
это ожидаемый результат, а не ошибка модели; в выводах это отдельно обсуждается.
