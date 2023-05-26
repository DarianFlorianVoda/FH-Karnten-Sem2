FROM python:3.11

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY meow/app.py .

CMD python meow/app.py