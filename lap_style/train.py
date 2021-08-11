import os
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)

from lap_layers import get_vgg19_encoder, get_decoder
from lap_dataset import TFDataLoader
from lap_losses import prepare_draft_losses


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
        optimizer,
        losses
    ):
        super(DraftNetModel, self).compile()
        self.optimizer = optimizer

        self.content_loss = losses[0]
        self.style_loss = losses[1]
        self.identity_loss_1 = losses[2]
        self.identity_loss_2 = losses[3]
        self.style_remd_loss = losses[4]
        self.content_relt_loss = losses[5]


    def train_step(self, data):
        ci, si = data[0], data[1]
        cF = self.encoder(ci, training=False)
        sF = self.encoder(si, training=False)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.decoder.trainable_variables)
            # draft output
            stylized = self.decoder([
                cF[3], sF[3], cF[2], sF[2], cF[1], sF[1]
            ], training=True)

            # stylized(content, content)
            # using identity loss
            Icc = self.decoder([
                cF[3], cF[3], cF[2], cF[2], cF[1], cF[1]
            ], training=True)

            # Featured stylized(content, content)
            # using identity loss
            Fcc = self.encoder(Icc, training=False)

            # prepare backward
            tF = self.encoder(stylized, training=False)

            # lp, content loss
            loss_c = self.content_loss(*(cF+tF))

            # lm, style loss
            loss_s = self.style_loss(*(sF+tF))

            # identity loss 1
            loss_il1 = self.identity_loss_1(ci, Icc)

            # identity loss 2s
            loss_il2 = self.identity_loss_2(*(cF+Fcc))

            # lr, style rEMD loss
            loss_r = self.style_remd_loss(sF[2], sF[3], tF[2], tF[3])

            # lss, content relative loss
            loss_ss = self.content_relt_loss(cF[2], cF[3], tF[2], tF[3])

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
        cF = self.encoder(ci, training=False)
        sF = self.encoder(si, training=False)

        stylized = self.decoder([
            cF[3], sF[3], cF[2], sF[2], cF[1], sF[1]
        ], training=False)

        # prepare backward
        tF = self.encoder(stylized, training=False)

        # lp, content loss
        loss_c = self.content_loss(*(cF+tF))

        # lm, style loss
        loss_s = self.style_loss(*(sF+tF))
        
        # identity loss 1
        # content vs stylized(content, content)
        Icc = self.decoder([
            cF[3], cF[3], cF[2], cF[2], cF[1], cF[1]
        ], training=False)
        loss_il1 = self.identity_loss_1(ci, Icc)

        # identity loss 2
        # Featured content cs Featured stylized(content, content)
        Fcc = self.encoder(Icc, training=False)
        loss_il2 = self.identity_loss_2(*(cF+Fcc))

        # lr, style rEMD loss
        loss_r = self.style_remd_loss(sF[2], sF[3], tF[2], tF[3])

        # lss, content relative loss
        loss_ss = self.content_relt_loss(cF[2], cF[3], tF[2], tF[3])

        #l draft
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


def main():
    train_idg = TFDataLoader(coco2017_path, style_path, 5, 'Train', limits=20000).gen_ds()
    val_idg = TFDataLoader(v_coco2017_path, style_path, 5, 'Val', shuffle=False).gen_ds()

    trainer_model = DraftNetModel()
    trainer_model.compile(
        optimizer=Adam(lr=1e-04),
        losses=prepare_draft_losses()
    )

    weight_name = 'draft.h5'
    # コールバック
    ## val_lossが更新されたときだけmodelを保存
    mc_cb = ModelCheckpoint(weight_name, verbose=1, save_best_only=True, monitor='val_draft_loss', save_weights_only=True, mode='min')
    ## 学習が停滞したとき、学習率を0.2倍に
    rl_cb = ReduceLROnPlateau(monitor='draft_loss', factor=0.2, patience=5, verbose=1)
    ## 学習が進まなくなったら、強制的に学習終了
    es_cb = EarlyStopping(monitor='draft_loss', patience=30, verbose=1)
    ## TensorBoardにログを書き込む
    tb_cb = TensorBoard(log_dir='tb_logs')

    trainer_model.fit(
        train_idg, validation_data=val_idg,
        epochs=100,
        callbacks=[mc_cb, rl_cb, es_cb, tb_cb])


if __name__ == '__main__':
    main()
