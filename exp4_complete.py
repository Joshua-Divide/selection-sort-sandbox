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

# Complete hyperparameter study on a frozen ImageNet MobileNetV2 base
pre=keras.applications.mobilenet_v2.preprocess_input
base=keras.applications.MobileNetV2(weights='imagenet',include_top=False,input_shape=(32,32,3),pooling='avg');base.trainable=False
def feats(x): return base.predict(pre(x.astype('float32')),batch_size=256,verbose=0)
ft,fv,fte=feats(x_train),feats(x_val),feats(x_test)
def head(units=128,opt='Adam',lr=.001):
 m=keras.Sequential([layers.Input((ft.shape[1],)),layers.Dense(units,activation='relu'),layers.Dropout(.2),layers.Dense(10,activation='softmax')])
 o=keras.optimizers.Adam(lr) if opt=='Adam' else keras.optimizers.SGD(lr,momentum=.9)
 m.compile(optimizer=o,loss='sparse_categorical_crossentropy',metrics=['accuracy']);return m
def run(label,lr=.001,batch=32,epochs=10,opt='Adam',units=128):
 m=head(units,opt,lr);t=time.perf_counter();h=m.fit(ft,y_train,validation_data=(fv,y_val),epochs=epochs,batch_size=batch,verbose=0);sec=time.perf_counter()-t
 _,acc=m.evaluate(fte,y_test,batch_size=256,verbose=0)
 r={'Setting':label,'Learning Rate':lr,'Batch Size':batch,'Epochs':epochs,'Optimizer':opt,'Dense Units':units,'Frozen Layers':'All','Best Validation Accuracy':float(max(h.history['val_accuracy'])),'Test Accuracy':float(acc),'Training Time (s)':float(sec)}
 del m;gc.collect();return r
hyp=[run('Baseline'),run('Learning rate 0.0001',lr=.0001),run('Batch size 16',batch=16),run('Batch size 64',batch=64),run('Epochs 20',epochs=20),run('Optimizer SGD',opt='SGD'),run('Dense units 256',units=256)]
pd.DataFrame(hyp).to_csv(OUT/'hyperparameter_complete.csv',index=False)

# Controlled architecture benchmark: same stratified 10k/2k/2k split for every model
bx,_,by,_=train_test_split(x_train,y_train,train_size=10000,random_state=SEED,stratify=y_train)
bv,_,bvy,_=train_test_split(x_val,y_val,train_size=2000,random_state=SEED,stratify=y_val)
bt,_,bty,_=train_test_split(x_test,y_test,train_size=2000,random_state=SEED,stratify=y_test)
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
 tf.keras.backend.clear_session();m=builder();m.compile(optimizer=keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy']);t=time.perf_counter();m.fit(xtr,by,validation_data=(xva,bvy),epochs=5,batch_size=64,verbose=2);sec=time.perf_counter()-t;_,acc=m.evaluate(xte,bty,batch_size=256,verbose=0);r={'Model':name,'Parameters':int(m.count_params()),'Accuracy':float(acc),'Accuracy (%)':float(acc*100),'Training Time (s)':float(sec)};del m;gc.collect();return r
arch=[scratch('LeNet-5',lenet),scratch('AlexNet',alexnet),scratch('GoogleNet',googlenet)]
def transfer(builder,prep,name,last):
 tf.keras.backend.clear_session();b=builder(weights='imagenet',include_top=False,input_shape=(32,32,3),pooling='avg');b.trainable=False;i=keras.Input((32,32,3));x=prep(i);x=b(x,training=False);x=layers.Dense(128,activation='relu')(x);m=keras.Model(i,layers.Dense(10,activation='softmax')(x));m.compile(optimizer=keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy']);t=time.perf_counter();m.fit(bx,by,validation_data=(bv,bvy),epochs=5,batch_size=64,verbose=2);b.trainable=True
 for_l= b.layers[:-last]
 for layer in for_l: layer.trainable=False
 for layer in b.layers[-last:]:
  if isinstance(layer,layers.BatchNormalization): layer.trainable=False
 m.compile(optimizer=keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy']);m.fit(bx,by,validation_data=(bv,bvy),epochs=1,batch_size=64,verbose=2);sec=time.perf_counter()-t;_,acc=m.evaluate(bt,bty,batch_size=256,verbose=0);r={'Model':name,'Parameters':int(m.count_params()),'Accuracy':float(acc),'Accuracy (%)':float(acc*100),'Training Time (s)':float(sec)};del m,b;gc.collect();return r
arch.append(transfer(keras.applications.VGG16,keras.applications.vgg16.preprocess_input,'VGG16',4))
arch.append(transfer(keras.applications.ResNet50,keras.applications.resnet50.preprocess_input,'ResNet50',10))
adf=pd.DataFrame(arch);adf.to_csv(OUT/'architecture_comparison.csv',index=False)
# Explicit additional exercise tables
hdf=pd.DataFrame(hyp)
hdf[hdf['Setting'].isin(['Baseline','Optimizer SGD'])].to_csv(OUT/'adam_vs_sgd.csv',index=False)
with open(OUT/'complete_results.json','w') as f: json.dump({'hyperparameter_study':hyp,'architecture_comparison':arch},f,indent=2)
print(adf);print(hdf)
