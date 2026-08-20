import time,json,gc,random
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
SEED=42
random.seed(SEED);np.random.seed(SEED);tf.random.set_seed(SEED)
OUT=Path('outputs_complete');OUT.mkdir(exist_ok=True)
(x_all,y_all),(x_test,y_test)=keras.datasets.cifar10.load_data();y_all=y_all.ravel();y_test=y_test.ravel()
x_train,x_val,y_train,y_val=train_test_split(x_all,y_all,test_size=5000,random_state=SEED,stratify=y_all)

def subset(x,y,n):
    xx,_,yy,_=train_test_split(x,y,train_size=n,random_state=SEED,stratify=y)
    return xx,yy

# Hyperparameter study on a fixed MobileNetV2 transfer-learning feature subset.
hx,hy=subset(x_train,y_train,10000); hv,hvy=subset(x_val,y_val,2000); ht,hty=subset(x_test,y_test,2000)
mpre=keras.applications.mobilenet_v2.preprocess_input
mbase=keras.applications.MobileNetV2(weights='imagenet',include_top=False,input_shape=(32,32,3),pooling='avg');mbase.trainable=False
def mf(x): return mbase.predict(mpre(x.astype('float32')),batch_size=256,verbose=0)
hf,hvf,htf=mf(hx),mf(hv),mf(ht)
def make_head(units=128,opt='Adam',lr=.001):
    m=keras.Sequential([layers.Input((hf.shape[1],)),layers.Dense(units,activation='relu'),layers.Dropout(.2),layers.Dense(10,activation='softmax')])
    o=keras.optimizers.Adam(lr) if opt=='Adam' else keras.optimizers.SGD(lr,momentum=.9)
    m.compile(optimizer=o,loss='sparse_categorical_crossentropy',metrics=['accuracy']);return m
def hrun(label,lr=.001,batch=32,epochs=10,opt='Adam',units=128):
    m=make_head(units,opt,lr);t=time.perf_counter();h=m.fit(hf,hy,validation_data=(hvf,hvy),epochs=epochs,batch_size=batch,verbose=0);sec=time.perf_counter()-t
    _,acc=m.evaluate(htf,hty,batch_size=256,verbose=0)
    r={'Setting':label,'Learning Rate':lr,'Batch Size':batch,'Epochs':epochs,'Optimizer':opt,'Dense Units':units,'Frozen Layers':'All','Best Validation Accuracy':float(max(h.history['val_accuracy'])),'Test Accuracy':float(acc),'Training Time (s)':float(sec)}
    del m;gc.collect();return r
hyp=[hrun('Baseline'),hrun('Learning rate 0.0001',lr=.0001),hrun('Batch size 16',batch=16),hrun('Batch size 64',batch=64),hrun('Epochs 20',epochs=20),hrun('Optimizer SGD',opt='SGD'),hrun('Dense units 256',units=256)]
pd.DataFrame(hyp).to_csv(OUT/'hyperparameter_complete.csv',index=False)
del mbase,hf,hvf,htf;gc.collect();tf.keras.backend.clear_session()

# Controlled architecture comparison on one fixed stratified subset.
bx,by=subset(x_train,y_train,1000);bv,bvy=subset(x_val,y_val,500);bt,bty=subset(x_test,y_test,1000)
xtr=bx.astype('float32')/255.;xva=bv.astype('float32')/255.;xte=bt.astype('float32')/255.
def lenet():
    i=keras.Input((32,32,3));x=layers.Conv2D(6,5,activation='tanh')(i);x=layers.AveragePooling2D(2)(x);x=layers.Conv2D(16,5,activation='tanh')(x);x=layers.AveragePooling2D(2)(x);x=layers.Flatten()(x);x=layers.Dense(120,activation='tanh')(x);x=layers.Dense(84,activation='tanh')(x);return keras.Model(i,layers.Dense(10,activation='softmax')(x))
def alexnet():
    i=keras.Input((32,32,3));x=layers.Conv2D(64,3,padding='same',activation='relu')(i);x=layers.MaxPooling2D(2)(x);x=layers.Conv2D(192,3,padding='same',activation='relu')(x);x=layers.MaxPooling2D(2)(x);x=layers.Conv2D(384,3,padding='same',activation='relu')(x);x=layers.Conv2D(256,3,padding='same',activation='relu')(x);x=layers.Conv2D(256,3,padding='same',activation='relu')(x);x=layers.MaxPooling2D(2)(x);x=layers.GlobalAveragePooling2D()(x);x=layers.Dense(512,activation='relu')(x);x=layers.Dropout(.5)(x);return keras.Model(i,layers.Dense(10,activation='softmax')(x))
def inc(x,f):
    p1=layers.Conv2D(f,1,padding='same',activation='relu')(x);p2=layers.Conv2D(f,1,padding='same',activation='relu')(x);p2=layers.Conv2D(f,3,padding='same',activation='relu')(p2);p3=layers.Conv2D(f//2,1,padding='same',activation='relu')(x);p3=layers.Conv2D(f//2,5,padding='same',activation='relu')(p3);p4=layers.MaxPooling2D(3,strides=1,padding='same')(x);p4=layers.Conv2D(f//2,1,padding='same',activation='relu')(p4);return layers.Concatenate()([p1,p2,p3,p4])
def googlenet():
    i=keras.Input((32,32,3));x=layers.Conv2D(64,3,padding='same',activation='relu')(i);x=layers.MaxPooling2D(2)(x);x=inc(x,64);x=inc(x,96);x=layers.MaxPooling2D(2)(x);x=inc(x,128);x=layers.GlobalAveragePooling2D()(x);x=layers.Dropout(.3)(x);return keras.Model(i,layers.Dense(10,activation='softmax')(x))
def scratch(name,builder):
    tf.keras.backend.clear_session();m=builder();m.compile(optimizer=keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy']);t=time.perf_counter();m.fit(xtr,by,validation_data=(xva,bvy),epochs=1,batch_size=64,verbose=2);sec=time.perf_counter()-t;_,acc=m.evaluate(xte,bty,batch_size=256,verbose=0);r={'Model':name,'Parameters':int(m.count_params()),'Accuracy':float(acc),'Accuracy (%)':float(acc*100),'Training Time (s)':float(sec)};del m;gc.collect();return r
arch=[scratch('LeNet-5',lenet),scratch('AlexNet',alexnet),scratch('GoogleNet',googlenet)]

def transfer_features(builder,prep,name):
    tf.keras.backend.clear_session();b=builder(weights='imagenet',include_top=False,input_shape=(32,32,3),pooling='avg');b.trainable=False
    t=time.perf_counter();ftr=b.predict(prep(bx.astype('float32')),batch_size=128,verbose=0);fva=b.predict(prep(bv.astype('float32')),batch_size=128,verbose=0);fte=b.predict(prep(bt.astype('float32')),batch_size=128,verbose=0)
    h=keras.Sequential([layers.Input((ftr.shape[1],)),layers.Dense(128,activation='relu'),layers.Dense(10,activation='softmax')]);h.compile(optimizer=keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy']);h.fit(ftr,by,validation_data=(fva,bvy),epochs=3,batch_size=64,verbose=0);sec=time.perf_counter()-t;_,acc=h.evaluate(fte,bty,batch_size=256,verbose=0)
    params=int(b.count_params()+h.count_params());r={'Model':name,'Parameters':params,'Accuracy':float(acc),'Accuracy (%)':float(acc*100),'Training Time (s)':float(sec)};del b,h,ftr,fva,fte;gc.collect();return r
arch.append(transfer_features(keras.applications.VGG16,keras.applications.vgg16.preprocess_input,'VGG16'))
arch.append(transfer_features(keras.applications.ResNet50,keras.applications.resnet50.preprocess_input,'ResNet50'))
pd.DataFrame(arch).to_csv(OUT/'architecture_comparison.csv',index=False)
pd.DataFrame(hyp)[pd.DataFrame(hyp)['Setting'].isin(['Baseline','Optimizer SGD'])].to_csv(OUT/'adam_vs_sgd.csv',index=False)
with open(OUT/'complete_results.json','w') as f: json.dump({'benchmark_training_images':1000,'benchmark_validation_images':500,'benchmark_testing_images':1000,'hyperparameter_training_images':10000,'hyperparameter_validation_images':2000,'hyperparameter_testing_images':2000,'benchmark_protocol':'VGG16 and ResNet50 use frozen ImageNet feature extraction with a new classifier; LeNet-5, AlexNet and GoogleNet are trained from scratch for one epoch.','hyperparameter_study':hyp,'architecture_comparison':arch},f,indent=2)
print(pd.DataFrame(arch));print(pd.DataFrame(hyp))
