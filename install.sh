git submodule update --init
cd maskrcnn && python setup.py install
cd .. && python setup.py install
pip install -r requirements.txt

mkdir models
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1USddPiSrD9DWIGzlTZ4xGkhZ11GAgrvR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1USddPiSrD9DWIGzlTZ4xGkhZ11GAgrvR" -O models/sd_maskrcnn.h5 && rm cookies.txt
