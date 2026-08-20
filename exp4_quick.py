import urllib.request
url='https://raw.githubusercontent.com/Joshua-Divide/selection-sort-sandbox/exp4-complete-20260820/exp4_complete.py'
src=urllib.request.urlopen(url).read().decode('utf-8')
src=src.replace('train_size=4000','train_size=1000')
src=src.replace('train_size=1000,random_state=SEED,stratify=y_val','train_size=500,random_state=SEED,stratify=y_val')
src=src.replace("epochs=3,batch_size=64,verbose=2","epochs=1,batch_size=64,verbose=2")
src=src.replace("'benchmark_training_images':4000,'benchmark_validation_images':1000", "'benchmark_training_images':1000,'benchmark_validation_images':500")
exec(compile(src,'exp4_complete_quick.py','exec'))
