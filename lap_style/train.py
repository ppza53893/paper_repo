import os
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from datetime import datetime

from lap_layers import get_vgg19_encoder, get_decoder
from lap_dataset import TFDataLoader
from lap_losses import *


current_dir = os.path.split(os.path.abspath(__file__))[0]
coco2017_path = os.path.join(current_dir, 'train2017')
v_coco2017_path = os.path.join(current_dir, 'val2017')
style_path = os.path.join(current_dir, 'wikiart/Symbolism/albert-pinkham-ryder_flying-dutchman-1887.jpg')


class DraftNetModel(Model):
    def __init__(self):
        super(DraftNetModel, self).__init__()
        self.encoder = get_vgg19_encoder()
        self.decoder = get_decoder()
    
    def compile(
        self,
        optimizer
    ):
        super(DraftNetModel, self).compile()
        self.optimizer = optimizer

        self.content_loss = content_loss
        self.style_loss = style_loss
        self.identity_loss_1 = identity_loss
        #self.identity_loss_2 = content_loss
        self.style_remd_loss = style_remd_loss
        self.content_relt_loss = content_relt_loss

        self.vgg_encoder = lambda x: self.encoder(x, training=False)
        #self.train_step = tf.function(experimental_relax_shapes=True)(self.train_step)

    def train_step(self, data):
        ci, si = data[0], data[1]
        cF = self.vgg_encoder(ci)
        sF = self.vgg_encoder(si)

        with tf.GradientTape() as tape:

            # draft output
            stylized = self.decoder([
                cF[3], sF[3], cF[2], sF[2], cF[1], sF[1]
            ], training=True)

            # stylized(content, content)
            # using identity loss
            Icc = self.decoder([
                cF[3], cF[3], cF[2], cF[2], cF[1], cF[1]
            ], training=True)

            with tape.stop_recording():
                # Featured stylized(content, content)
                # using identity loss
                Fcc = self.vgg_encoder(Icc)

                # prepare backward
                tF = self.vgg_encoder(stylized)

            # lp, content loss
            loss_c = self.content_loss(cF[0], tF[0])+self.content_loss(cF[1], tF[1])+\
                self.content_loss(cF[2], tF[2])+self.content_loss(cF[3], tF[3])+\
                    self.content_loss(cF[4], tF[4])

            # lm, style loss
            loss_s = self.style_loss(sF[0], tF[0])+self.style_loss(sF[1], tF[1])+\
                self.style_loss(sF[2], tF[2])+self.style_loss(sF[3], tF[3])+\
                    self.style_loss(sF[4], tF[4])

            # identity loss 1
            loss_il1 = self.identity_loss_1(ci, Icc)

            # identity loss 2s
            loss_il2 = self.content_loss(cF[0], Fcc[0])+self.content_loss(cF[1], Fcc[1])+\
                self.content_loss(cF[2], Fcc[2])+self.content_loss(cF[3], Fcc[3])+\
                    self.content_loss(cF[4], Fcc[4])

            # lr, style rEMD loss
            loss_r = self.style_remd_loss(sF[2], tF[2]) + self.style_remd_loss(sF[3], tF[3])

            # lss, content relative loss
            loss_ss = self.content_relt_loss(cF[2], tF[2]) + self.content_relt_loss(cF[3], tF[3])

            #loss
            loss = loss_c*1. + loss_s*3. + loss_il1*50. + loss_il2*1. + loss_r*10. + loss_ss*16.
        # get gradients
        grad = tape.gradient(loss, self.decoder.trainable_variables)

        # update the weights
        self.optimizer.apply_gradients(zip(grad, self.decoder.trainable_variables))

        return {
            "draft_loss": loss,
            "style_loss": loss_s,
            "style_rEMD_loss": loss_r,
            "content_loss": loss_c,
            "content_relt_loss": loss_ss,
            "identity_loss_1": loss_il1,
            "identity_loss_2": loss_il2
            }

    def test_step(self, data):
        ci, si = data[0], data[1]
        cF = self.vgg_encoder(ci)
        sF = self.vgg_encoder(si)

        stylized = self.decoder([
            cF[3], sF[3], cF[2], sF[2], cF[1], sF[1]
        ], training=False)

        # stylized(content, content)
        # using identity loss
        Icc = self.decoder([
            cF[3], cF[3], cF[2], cF[2], cF[1], cF[1]
        ], training=False)

        # Featured stylized(content, content)
        # using identity loss
        Fcc = self.vgg_encoder(Icc)

        # prepare backward
        tF = self.vgg_encoder(stylized)

        loss_c = self.content_loss(cF[0], tF[0])+self.content_loss(cF[1], tF[1])+\
            self.content_loss(cF[2], tF[2])+self.content_loss(cF[3], tF[3])+\
                self.content_loss(cF[4], tF[4])

        # lm, style loss
        # graph: 3
        loss_s = self.style_loss(sF[0], tF[0])+self.style_loss(sF[1], tF[1])+\
            self.style_loss(sF[2], tF[2])+self.style_loss(sF[3], tF[3])+\
                self.style_loss(sF[4], tF[4])

        # identity loss 1
        loss_il1 = self.identity_loss_1(ci, Icc)

        # identity loss 2s
        #loss_il2 = self.identity_loss_2(*(cF+Fcc))
        loss_il2 = self.content_loss(cF[0], Fcc[0])+self.content_loss(cF[1], Fcc[1])+\
            self.content_loss(cF[2], Fcc[2])+self.content_loss(cF[3], Fcc[3])+\
                self.content_loss(cF[4], Fcc[4])

        # lr, style rEMD loss
        # loss_r = self.style_remd_loss(sF[2], sF[3], tF[2], tF[3])
        loss_r = self.style_remd_loss(sF[2], tF[2]) + self.style_remd_loss(sF[3], tF[3])

        # lss, content relative loss
        # loss_ss = self.content_relt_loss(cF[2], cF[3], tF[2], tF[3])
        loss_ss = self.content_relt_loss(cF[2], tF[2]) + self.content_relt_loss(cF[3], tF[3])

        #loss
        loss = loss_c*1. + loss_s*3. + loss_il1*50. + loss_il2*1. + loss_r*10. + loss_ss*16.

        return {
            "draft_loss": loss,
            "style_loss": loss_s,
            "style_rEMD_loss": loss_r,
            "content_loss": loss_c,
            "content_relt_loss": loss_ss,
            "identity_loss_1": loss_il1,
            "identity_loss_2": loss_il2
            }

    def predict_step(self, data):
        ci, si = data[0], data[1]
        cF = self.encoder(ci, training=False)
        sF = self.encoder(si, training=False)

        return self.decoder([
            cF[3], sF[3], cF[2], sF[2], cF[1], sF[1]
        ], training=False)

    def call(self, data, training=False, mask=None):
        return self.predict_step(data)


def main_seq():
    trainer = TFDataLoader(coco2017_path, style_path, 5, 'Train', limits=None).gen_ds()
    validation = TFDataLoader(v_coco2017_path, style_path, 5, 'Val', shuffle=False).gen_ds()

    trainer_model = DraftNetModel()
    trainer_model.compile(
        optimizer=Adam(lr=1e-04)
    )

    weight_name = 'draft_model.h5'
    # コールバック
    ## val_lossが更新されたときだけmodelを保存
    mc_cb = ModelCheckpoint(weight_name, verbose=1, save_best_only=True, monitor='val_draft_loss', save_weights_only=True, mode='min')
    ## 学習が停滞したとき、学習率を0.2倍に
    rl_cb = ReduceLROnPlateau(monitor='draft_loss', factor=0.2, patience=5, verbose=1)
    ## 学習が進まなくなったら、強制的に学習終了
    es_cb = EarlyStopping(monitor='draft_loss', patience=30, verbose=1)
    ## TensorBoardにログを書き込む
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_cb = TensorBoard(log_dir='tfr_train/{}'.format(stamp))

    trainer_model.fit(
        trainer, validation_data=validation,
        epochs=1000,
        callbacks=[mc_cb, rl_cb, es_cb, tb_cb])


if __name__ == '__main__':
    main_seq()
