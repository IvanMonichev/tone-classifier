# Tone Classifier

## Описание
Этот проект представляет собой классификатор тональности текста на основе различных моделей машинного обучения.

## Требования
- Python 3.6 или выше

## Инструкция

1. Этот проект требует скачивания двух файлов в папку `content`. Чтобы это сделать, выполните следующую команду:

**Linux / MacOS / WSL**
```bash
mkdir -p content && wget -P content https://raw.githubusercontent.com/Gavroshe/RuTweetCorp/master/positive.csv https://raw.githubusercontent.com/Gavroshe/RuTweetCorp/master/negative.csv
```


2. Создайте виртуальное окружение в каталоге проекта:
```bash
python -m venv venv

```
3. Активируйте виртуальное окружение:
- Для Windows:
```bash
venv\Scripts\activate
```

- Для macOS и Linux:

```bash
source venv/bin/activate
```

4. Установите зависимости
```bash
pip install -r requirements.txt
```

5. Запуск

```bash
python main.py
```