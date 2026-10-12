### Что это?
Этот репозиторий является неким "сборником" учебных материалов касательно работы с фоновыми задачами, асинхронными приложениями и event-driven архитектурами в основе которых лежит RabbitMQ.

### Базовая задача


### Методы решения
Вся "бизнес" логика написана в файле `recognition.py`, по сути там эмулируется некая полезная работа для задачи детекции объектов на изображении

```python
result = model.predict(image, conf=conf, verbose=False)[0]
```
По сути всё сводится к вот этой строчке. Но это не важно, тут может быть всё, что угодно. Важная не сама задача, а её "оркестрация".

#### Классический "в лоб"
Тут всё сводится к одному эндпоинту
Проблема тут в том. Что вычисления блокируют event-loop и мы масштабируем нашу ML часть непосредственно вместе с web частью (хотя и текущий пример недостаточно сложен, чтобы эту проблему подсветить)

```python
@app.post("/predict")
async def predict_image(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    return predict(model, image)

```

#### C использованием Celery
Если вы когда-либо работали с Celery, тут тоже всё абсолютно просто и прозрачно.  

```python
@celery.task
def predict_task(image_path: str):
    global model 

    if model is None:
        model = load_model()

    return predict(model, image_path)

## вызываем (правильнее сказать планируем) так
predict_task.delay(path)
```

Единственная явная проблема - наличие global. Я честно сказать лучше решения, чем https://dishka.readthedocs.io/en/stable/integrations/celery.html не знаю. Хотя, можно конечно использовать и singleton.

#### "Outbox-Inbox like" 
Файл `outbox_inbox.py`. Решаем проблему двойной записи (dual write): если сначала записать задачу в БД, а потом отправить сообщение в брокер, то между этими шагами процесс может упасть, и задача навсегда останется в `PENDING`. Или наоборот: сообщение уйдёт, а транзакция откатится.

1. При создании задачи сохраняем её в таблицу `tasks`, а сообщение для брокера в таблицу `outbox`, в **одной транзакции**. Либо записалось и то и другое, либо ничего.
```python
with db:
    db.execute("INSERT INTO tasks (id, image_path, status) VALUES (?, ?, 'PENDING')", ...)
    db.execute("INSERT INTO outbox (message_id, payload) VALUES (?, ?)", ...)
```
2. Отдельный процесс relay раз в 0.5 с выбирает из `outbox` неотправленные строки, публикует их в RabbitMQ и после подтверждения от брокера помечает `published = 1`.
3. Relay может упасть между публикацией и пометкой, тогда сообщение уйдёт повторно. Гарантия at-least-once, поэтому воркер должен быть идемпотентным: id обработанных сообщений он пишет в таблицу `inbox` в одной транзакции с результатом и дубликаты пропускает (но ack-ает!).

Минусы поллинга: постоянная нагрузка на БД, задержка до 0.5 с, два relay без `SELECT ... FOR UPDATE SKIP LOCKED` отправят одно и то же (а в SQLite его нет).

#### Outbox через журнал транзакций (CDC)
Файл `outbox_cdc.py`. Совет из [статьи](https://thecodinggopher.substack.com/p/the-transactional-outbox-pattern): «поллинг, пока не заболит, потом CDC». Каждый закоммиченный `INSERT INTO outbox` и так попадает в журнал транзакций Postgres (WAL), так что вместо опроса таблицы можно читать журнал.

- Postgres запускается с `wal_level=logical` и плагином `wal2json`, который превращает записи WAL в JSON (см. `docker-compose.yml`).
- Создаётся слот логической репликации. Postgres хранит WAL, пока слот его не прочитал, а в самом слоте лежит позиция чтения (LSN). Поэтому колонка `published` не нужна.
- Relay читает из слота только вставки в `public.outbox`, публикует в RabbitMQ с publisher confirms и только после подтверждения сдвигает позицию слота:
```python
channel.basic_publish(...)                          # брокер подтвердил
msg.cursor.send_feedback(flush_lsn=msg.data_start)  # теперь слот можно двигать
```
- Строки `outbox` после коммита relay уже не нужны (событие лежит в WAL), их можно чистить.

Почему не `pg_notify`: NOTIFY отправляется при коммите, но не хранится. Если relay в этот момент не слушает, уведомление потеряно. Он годится только как сигнал «проснись» для поллинга, но не как журнал.

Подводные камни: слот, который никто не читает, копит WAL (поэтому `max_slot_wal_keep_size`); пересозданный слот не знает старых событий; порядок сохраняется, пока relay один, а дальше надо партиционировать по `aggregate_id`.

```bash
docker compose up -d --build
uvicorn outbox_cdc:api --port 8000
python outbox_cdc.py relay
python outbox_cdc.py worker
```

#### FastStream пример

