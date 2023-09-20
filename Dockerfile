FROM python:3.11.4

WORKDIR /usr/src/app

COPY requirements_docker.txt ./
RUN pip install --no-cache-dir -r requirements_docker.txt

COPY . .

CMD [ "chainlit", "run","model.py","-w" ]

#docker run -p 8000:8000 -it --rm --name bot chatbot