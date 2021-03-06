# torchvisionBGremoval

**TLDR**

**ОБУЧЕН НАХОДИТЬ ТОЛЬКО ИЗОБРАЖЕНИЯ ЛЮДЕЙ**

[BGRemovalApp.py](BGRemovalApp.py) - удаляет фон у изображения, путь до которого подается на вход

[8 epoch params archive](https://github.com/evgen32cd32/torchvisionBGremoval/tree/main/8%20epoch%20params%20archive) - разбитый zip архив файла с параметрами модели

[flask_app.py](flask_app.py) - код для телеграм-бота [@evgensbgremovalbot](https://t.me/evgensbgremovalbot) на heroku. Есть проблемы с использованием памяти, если он начинает потреблять больше ~2Гб, heroku его перезагружает. Также если им некоторое время не пользоваться - сервис засыпает, требуется время, чтобы подняться.


## Датасет
Это решение задачи по удалению фона у центрального объекта изображения.

В качестве датасета был взят [COCO](https://cocodataset.org/#home) train и val 2014 datasets.

Для выбора центральной фигуры для всех non-crowd аннотаций я считаю 'barycenterWeight' - площадь маски, деленная на четвертую степени расстояния от центра boundary box до центра картинки.

После безуспешных попыток оптимизации путем создания батчей и увеличения num_workers я уменьшил выборку еще двумя ограничениями: объект должен быть класса person и занимать не менее 40% площади. В итоге мои выборки:

- Train dataset: 2088
- Test dataset: 1100

## Нейросеть
Решение строится на сверточных нейроных сетях (RNN) и алгоритме U-net, развертывающем изображение в маску. Пример [реализации](https://github.com/YunanWu2168/Background-removal-using-deep-learning). Также существует еще методология ResNet - использование остаточных значений на следующих слоях сети.

Мое решение основано на [туториале](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), используется подготовленная **maskrcnn_resnet50_fpn** и тренируется только последний слой head (fine tuning). Также используются [утилиты](https://github.com/pytorch/vision/tree/main/references/detection).

Так как мне нужна только одна маска, я решил написать свой класс датасета. У меня так и не получилось подогнать вывод targets для скрипта evaluate() из [engine.py](engine.py), я в итоге унаследовал torchvision.datasets.CocoDetection (это было ошибкой).

Тренировал скриптом [BGRemovalTrain.py](BGRemovalTrain.py) на windows 10 с Geforce GTX 1050 TI. [Лог](/temp/TrainLog.txt).
Как выяснилось - evaluate() тащил api coco из trainset.coco и валидировал для всех масок, а моя нейросеть тренировалась только на одной.
Попытки оттестировать evaluate() можно видеть в [BGRemovalTest.py](BGRemovalTest.py).


Как альтернатива - решил использовать **maskrcnn_resnet50_fpn** как есть из коробки без дообучений, скрипт проверки [ABGR.py](ABGR.py).


## Метрики
Так как с evaluate() подружиться не получилось - решил самостоятельно рассчитать IoU - отношение площади пересечения масок к объединению.

В папке [metrics](https://github.com/evgen32cd32/torchvisionBGremoval/tree/main/metrics) рассчитанные метрики для каждого изображения из тестового и трейн наборов и для всех эпох с 0(до обучения) по 20. [ABGRtest_metrics.csv](ABGRtest_metrics.csv) - для альтернативного метода на тестовых данных.

Расчет велся соответствующими скриптами: [BGRemovalMetrics.py](BGRemovalMetrics.py) для тест, [BGRemovalMetrics_train.py](BGRemovalMetrics_train.py) для train и [ABGRmetrics.py](ABGRmetrics.py) для тест на альтернативном методе.

### Результаты:
![image](https://user-images.githubusercontent.com/25753000/167195181-3ff2ffb9-02dc-43b4-bac4-d25f9907b7f3.png)

Средние значения IoU для тест и трейн данных.

- Максимальное значение на тест сете на эпохе 8: 0.865928
- Значение на 20 эпохе: 0.864992
- Значение альтернативного метода на тест сете: 0.780053

**Замечание** Рассчеты метрик для 11-20 эпох были получены после подготовки скриптов для телеграм-бота, поэтому там крутится сейчас 20. Файлы параметров весят по 170 мегабайт, а так как бот все равно работает нестабильно, не вижу необходимости обновлять.

## Телеграм-бот
Была попытка запилить бота, сначала на [PythonAnywhere](https://www.pythonanywhere.com/), сейчас крутится на [Heroku](https://dashboard.heroku.com/). Скрипт [flask_app.py](flask_app.py).

Главная проблема - вылезает по памяти на втором изображении, попытки это исправить может потом еще будут, так что он пока работает в альфа режиме.

Как его неполноценная замена - [BGRemovalApp.py](BGRemovalApp.py) скрипт для домашнего использования.
