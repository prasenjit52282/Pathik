import tensorflow as tf
from .attention import Attention
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

class HybdModel:
    def __init__(self, image_shape=(64,64,3),seq_shape=(10,21)):
        self.image_shape=image_shape
        self.seq_shape=seq_shape
        self.initilize_InputLayers()
        self.build()

    def initilize_InputLayers(self):
        self.X_seq=tf.keras.layers.Input(shape=self.seq_shape)
        self.X_img=tf.keras.layers.Input(shape=self.image_shape)

    def LSTM(self,input_shape):
        inp=tf.keras.layers.Input(input_shape)
        net=tf.keras.layers.LSTM(units=21,return_sequences=False)(inp)
        out=tf.keras.layers.Dense(units=256,activation="relu")(net)
        att_model=tf.keras.Model(inputs=inp,outputs=out,name="LSTM")
        return att_model

    def ATT(self,input_shape):
        inp=tf.keras.layers.Input(input_shape)
        net=Attention((None,*input_shape))(inp)
        out=tf.keras.layers.Dense(units=256,activation="relu")(net)
        att_model=tf.keras.Model(inputs=inp,outputs=out,name="ATT")
        return att_model

    def CNN(self,input_shape):
        cnn_model=\
        tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,(5,5),(2,2),"same",activation="relu",input_shape=input_shape),
            tf.keras.layers.Conv2D(128,(3,3),(2,2),"same",activation="relu"),
            tf.keras.layers.Conv2D(256,(3,3),(2,2),"same",activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
        ],name='CNN')
        return cnn_model

    def ANN(self,input_shape):
        ann_model=\
        tf.keras.Sequential([
            tf.keras.layers.Dense(units=256,activation="relu",input_shape=input_shape),
            tf.keras.layers.Dense(units=64,activation="relu"),
            tf.keras.layers.Dense(units=1,activation="linear")
        ],name='ANN')
        return ann_model
    
    def build(self):
        self.model=None

    def train(self,X_train,speed_train,weight_path,batch_size=64,epochs=50,validation_split=0.2):
         best_save=tf.keras.callbacks.ModelCheckpoint(filepath=weight_path,save_weights_only=True,monitor='val_mae',mode='max',save_best_only=True)
         self.training_logs=self.model.fit(x=X_train,y=speed_train,batch_size=batch_size,epochs=epochs,validation_split=validation_split,callbacks=[best_save])

    def evaluate(self,X_train,speed_train,X_test,speed_test):
        pred=self.model.predict(X_train,verbose=0)
        mae_train=mean_absolute_error(speed_train,pred)
        mse_train=mean_squared_error(speed_train,pred)
        r2_train=r2_score(speed_train,pred)

        pred=self.model.predict(X_test,verbose=0)
        mae_test=mean_absolute_error(speed_test,pred)
        mse_test=mean_squared_error(speed_test,pred)
        r2_test=r2_score(speed_test,pred)

        self.eval={'mae_train':mae_train,'mse_train':mse_train,'r2_train':r2_train,'mae_test':mae_test,'mse_test':mse_test,'r2_test':r2_test}
        return self.eval
    


class OurModel(HybdModel):
        def __init__(self,image_shape=(64,64,3),seq_shape=(10,21)):
            super(OurModel, self).__init__(image_shape,seq_shape)
            self.build()

        def build(self):
            cnn=self.CNN(self.image_shape)
            att=self.ATT(self.seq_shape)
            ann=self.ANN((512,))

            cnn_emb=cnn(self.X_img)
            att_emb=att(self.X_seq)
            net=tf.concat([cnn_emb,att_emb],axis=1)
            out=ann(net)

            self.model=tf.keras.Model(inputs=[self.X_seq,self.X_img],outputs=out,name='OurModel')
            self.model.compile(loss="mse",optimizer="adam",metrics=["mae","mse"])


class LstmModel(HybdModel):
        def __init__(self,image_shape=(64,64,3),seq_shape=(10,21)):
            super(OurModel, self).__init__(image_shape,seq_shape)
            self.build()

        def build(self):
            cnn=self.CNN(self.image_shape)
            lstm=self.LSTM(self.seq_shape)
            ann=self.ANN((512,))

            cnn_emb=cnn(self.X_img)
            lstm_emb=lstm(self.X_seq)
            net=tf.concat([cnn_emb,lstm_emb],axis=1)
            out=ann(net)

            self.model=tf.keras.Model(inputs=[self.X_seq,self.X_img],outputs=out,name='LstmModel')
            self.model.compile(loss="mse",optimizer="adam",metrics=["mae","mse"])


class SeqModel(HybdModel):
        def __init__(self,image_shape=(64,64,3),seq_shape=(10,21)):
            super(OurModel, self).__init__(image_shape,seq_shape)
            self.build()

        def build(self):
            att=self.ATT(self.seq_shape)
            ann=self.ANN((256,))

            att_emb=att(self.X_seq)
            out=ann(att_emb)

            self.model=tf.keras.Model(inputs=self.X_seq,outputs=out,name='SeqModel')
            self.model.compile(loss="mse",optimizer="adam",metrics=["mae","mse"])


class ImgModel(HybdModel):
        def __init__(self,image_shape=(64,64,3),seq_shape=(10,21)):
            super(OurModel, self).__init__(image_shape,seq_shape)
            self.build()

        def build(self):
            cnn=self.CNN(self.image_shape)
            ann=self.ANN((256,))

            cnn_emb=cnn(self.X_img)
            out=ann(cnn_emb)

            self.model=tf.keras.Model(inputs=self.X_img,outputs=out,name='ImgModel')
            self.model.compile(loss="mse",optimizer="adam",metrics=["mae","mse"])