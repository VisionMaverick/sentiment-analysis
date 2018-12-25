# sentiment-analysis
Please follow below instruction to build and deploy sentiment analysis application

Install
  Python
    https://realpython.com/installing-python/
    Add below environment variables in PATH if not available
        C:\Python36(Python root folder)
        C:\Python36\Scripts(Python Executables)
    Upgrading pip
        On Linux or macOS:
			$ sudo apt-get install python3-pip
			or
            $ pip3 install -U pip
        On Windows:
            $ python -m pip install -U pip
Create virtual environment named venv under project root folder(sentiment-analysis)
    Windows
        $ cd sentiment-analysis
        $ pip install virtualenv
        $ virtualenv venv
		
		--Activate Virtual environment
        $ venv\Scripts\activate
	Linux--
        $ cd sentiment-analysis		
        $ sudo pip3 install virtualenv	
		
		$ virtualenv -p python3 venv
		or 
        $ python3 -m venv venv
		
		--Activate Virtual environment
        $ source venv/bin/activate
		
    Download NLTK corpora
        $ python -m nltk.downloader all

Installing dependencies
    $ pip3 install -r requirements/requirements.txt
Running server
    $ venv\Scripts\python server.py runserver
Running test
    $ venv\Scripts\python server.py test
Invoking REST API
    http://127.0.0.1:8000/swagger/
    http://127.0.0.1:8000/api/v1/predictions/?data=I have no headache!



