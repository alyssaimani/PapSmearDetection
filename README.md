# InDRI (Intelligent Diagnosis of Radiology Images)

It is a computer-aided diagnosis of lung disease using image-based machine learning. 

## Step to INSTALL 

1. ```git clone https://github.com/mahmudisnan/indri.git```

2. create local env with ```python=3.10```

3. run command ```pip install -r requirements.txt```

4. rename ```.env.example``` to ```.env```

5. on .env file you need to set your config

```SECRET_KEY=YOUR_KEY

DB_NAME=YOUR_DB_NAME 

DB_USER=YOUR_DB_USER

DB_PASS=YOUR_DB_PASS```

 	5.1 for your SECRET_KEY run this command 

	5.2 python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())”

	5.3 paste the generated key to your .env file

6. run command ```python manage.py makemigrations```

7. run command ```python manage.py migrate```

8. Login to http://127.0.0.1:8000/

9. you need to create admin account by run this command

10. run command ```python manage.py createsuperuser```