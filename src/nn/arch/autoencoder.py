    #******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# autoencoder classes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#******************************************************************************


import keras
from keras.optimizers import Adam
from keras import objectives
from keras.layers import *
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.models import Model, save_model, load_model
import logging
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from functools import partial

#from helpers.upsampling import *
from ..stages import *
#from helpers.transposed_convolution import *
from ..helpers import *
from ..losses import *
from ..metrics import KLDivergence
from ..callbacks import LossHistory
from nn.arch.architecture import Network
import time
from ops import *
from resutils import *

def _sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]), mean=0.0, stddev=0.2)
    return z_mean + K.exp(z_log_sigma) * epsilon

def convert_shape(shape):
    out_shape = []
    for i in shape:
        try:
            out_shape.append(int(i))
        except:
            out_shape.append(None)
    return out_shape

#=====================================================================================
class Autoencoder(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, settings=None, **kwargs):
        self.init_func = "glorot_normal"
        self.feature_multiplier = 32
        self.surface_kernel_size = 4
        self.kernel_size = 2
        self.adam_epsilon = 1e-8 # 1e-3
        self.adam_learning_rate = 0.001
        self.adam_lr_decay = 0.005#1e-5
        self.pretrain_epochs = 1#3
        self.input_shape = kwargs.get("input_shape", (64, 64, 64, 1))
        self.loss = kwargs.get("loss", "mse")
        self.metrics = ["mae"]
        if not isinstance(self.loss, str):
            self.metrics = ["mse", "mae"]
        self.variational_ae = False
        self.kl_beta = 1e-5
        self.tensorflow_seed = kwargs.get("tensorflow_seed", 4)
        self.model = None
        tf.set_random_seed(self.tensorflow_seed)

    #---------------------------------------------------------------------------------
    def _init_optimizer(self, epochs=1):
        self.optimizer = Adam(lr=self.adam_learning_rate, epsilon=self.adam_epsilon, decay=self.adam_lr_decay)
        return self.optimizer

        #---------------------------------------------------------------------------------
    def _build_model(self):
        # ENCODER ################################################################################## 
        input_shape = self.input_shape
        pressure_input = Input(shape=input_shape)

        encoder_stages = StagedModel()
        encoder_stages.start(pressure_input)
        
        encoder_input = pressure_input
        # Conv #
        x = Conv3D(self.feature_multiplier * 1, self.surface_kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(encoder_input)
        #x = BatchNormalization(mode=1)(x)
        x = Activation('linear')(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 2, self.kernel_size, strides=(2, 2, 2),padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)
        
        # Conv #
        x = Conv3D(self.feature_multiplier * 4, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 8, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 16, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        x = Conv3D(self.feature_multiplier * 32, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        # Flatten #
        encoder_output_shape = convert_shape(x.shape[1:]) # (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
       
        if self.variational_ae:
            self._z_size = encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]#settings.ae.code_layer_size
            x = Flatten()(x)
            x = Dense(units=self._z_size)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.3)(x)
        
        # Stage 4 END #
        encoder_stages.end(x)

        if self.variational_ae:
            # the following will be added to the autoencoder only and can not be part of the stage models due to some weird keras issue
            z_mean_layer = Dense(units=self._z_size)
            z_log_sigma_layer = Dense(units=self._z_size)
            z_layer = Lambda(_sampling, output_shape=(self._z_size,))

        # DECODER ####################################################################################################
        x = Input(shape=(None, None, None, self.feature_multiplier * 32))
        
        if self.variational_ae:
            decoder_input = Input(shape=(self._z_size,))
            x = decoder_input
    
        # Stage 4 BEGIN #
        decoder_stages = StagedModel()
        decoder_stages.start(x)

        if self.variational_ae:
            x = Dense(units=encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3])(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.3)(x)
            x = Reshape(target_shape=encoder_output_shape)(x)
        
        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 16, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 8, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 4, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 2, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)


        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 1, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)        
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = decoder_stages.stage(x)
        
        # Deconv #
        x = Conv3DTranspose(input_shape[-1], self.surface_kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        decoder_output = Activation('linear')(x)
        
        # Stage 1 END #
        decoder_stages.end(decoder_output)

        self._decoder = decoder_stages.model #Model(input=decoder_input, output = decoder_output)
        self._decoder.name = "Decoder"

        # AUTOENCODER #########################################################################################
        
        # Build staged autoencoder
        self._stages = []
        for stage_num in range(1, len(encoder_stages.stages) + 1):
            stage_input = Input(shape=input_shape)
            x = encoder_stages[0:stage_num](stage_input)
            stage_output = decoder_stages[6-stage_num:6](x)
            stage_model = Model(inputs=stage_input, outputs=stage_output)
            stage_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            #make_layers_trainable(self._stages[0], False)
            self._stages.append(stage_model)

        # Autoencoder
        ae_input = Input(shape=input_shape)
        h = encoder_stages[0:6](ae_input)

        if self.variational_ae:
            z_mean = z_mean_layer(h)
            z_log_sigma = z_log_sigma_layer(h)
            z = z_layer([z_mean, z_log_sigma])
        else:
            z = h

        ae_output = decoder_stages[0:6](z)
        self.model = Model(inputs=ae_input, outputs=ae_output)

        if self.variational_ae:
            self.vae_loss = VAELoss(z_mean=z_mean, z_log_var=z_log_sigma, kl_beta = self.kl_beta, loss=MSE(1.0))
            self.metrics.append(KLDivergence(z_mean=z_mean, z_log_var=z_log_sigma))
            self.model.compile(loss=self.vae_loss, optimizer=self.optimizer, metrics=self.metrics)#['mae', self.kl_div])
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            
        self.model.name = "Autoencoder"

        # Encoder
        enc_input = Input(shape=input_shape)
        h = encoder_stages[0:6](enc_input)

        if self.variational_ae:
            z_mean = z_mean_layer(h)
            z_log_sigma = z_log_sigma_layer(h)
            z = z_layer([z_mean, z_log_sigma])
        else:
            z = h

        self._encoder = Model(enc_input, z)
        self._encoder.name = "Encoder"

    #---------------------------------------------------------------------------------
    def _compile_model(self):
        pass
        # self._encoder.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        # self._decoder.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        # self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    #---------------------------------------------------------------------------------
    def _train(self, epochs, **kwargs):
        # get the relevant arguments
        dataset = kwargs.get("dataset", None)
        assert dataset is not None, ("You must provide a dataset argument to autoencoder train")
        batch_size = kwargs.get("batch_size", 32) 
        augment = kwargs.get("augment", False)
        
        # pretrain if enabled
        if not kwargs.get("disable_pretrain", False):
            self._pretrain(dataset=dataset, epochs=self.pretrain_epochs, batch_size=batch_size, augment=augment)
        
        # do the actual training
        return self._train_full_model(dataset=dataset, epochs=epochs, batch_size=batch_size, augment=augment)

    #---------------------------------------------------------------------------------
    def _pretrain(self, dataset, epochs = 10, batch_size=128, plot_callback=None, augment=False, noise=False):
        """ 
        Train the autoencoder in stages of increasing size 
        :param epochs: Number of epochs of training
        :param x: The dataset fed as input to the autoencoder
        :param y: The real dataset, by which the autoencoder will be measured
        """
        hist = None
        callbacks = None
        if plot_callback is not None:
            loss_hist = LossHistory(plot_callback)
            callbacks = [loss_hist]

        if epochs == 0:
            return None
        try:
            print("\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("Autoencoder pretraining")
            train_generator = dataset.train.generator(batch_size=batch_size, augment=augment, noise=noise)
            train_steps_per_epoch = dataset.train.steps_per_epoch(batch_size=batch_size, augment=augment)
            val_generator = dataset.val.generator(batch_size=batch_size, augment=augment, noise=noise)
            val_steps = dataset.val.steps_per_epoch(batch_size=batch_size, augment=augment)
            # Train the autoencoder stages
            for index, stage in enumerate(self._stages):
                print("\n--------------------------------------------------------------------------------")
                print("Stage {}\n".format(index + 1))
                
                hist = stage.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    epochs=epochs,
                    max_queue_size=100,
                    callbacks=callbacks)
                    
        except KeyboardInterrupt:
            print("\n Interrupted by user")
        
        #self.model = stage
        return hist

    def encoder_output_shape(self, input_shape=None):
        if not None in self.input_shape:
            input_shape = self.input_shape
        assert input_shape is not None, ("You must provide an input shape for autoencoders with variable input sizes")
        dummy_input = np.expand_dims(np.zeros(input_shape), axis = 0)
        shape = self.encode(dummy_input, 1).shape[1:]
        return shape

    #---------------------------------------------------------------------------------
    def _train_full_model(self, dataset, epochs = 10, batch_size=128, plot_callback=None, augment=False, noise=False):
        hist = None
        callbacks = None
        if plot_callback is not None:
            loss_hist = LossHistory(plot_callback)
            callbacks = [loss_hist]

        if epochs == 0:
            return None
        try:
            print("\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("Autoencoder training")
            train_generator = dataset.train.generator(batch_size=batch_size, augment=augment, noise=noise)
            train_steps_per_epoch = dataset.train.steps_per_epoch(batch_size=batch_size, augment=augment)
            val_generator = dataset.val.generator(batch_size=batch_size, augment=augment, noise=noise)
            val_steps = dataset.val.steps_per_epoch(batch_size=batch_size, augment=augment)

            hist = self.model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=val_generator,
                validation_steps=val_steps,
                epochs=epochs,
                max_queue_size=100,
                callbacks=callbacks)
                
        except KeyboardInterrupt:
            print("\n Interrupted by user")
    
        return hist

    #---------------------------------------------------------------------------------
    def print_summary(self):
        print("Autoencoder")
        self.model.summary()
        print("Encoder")
        self._encoder.summary()
        print("Decoder")
        self._decoder.summary()

    #---------------------------------------------------------------------------------
    def load_model(self, path):
        print("Loading model from {}".format(path))
        #self._vae = load_model(path)
        if self.model is None:
            self._build_model()
        self.model.load_weights(path, by_name=True)
        # if self._vae.to_json() != self._model.to_json():
        #     print("[WARNING] Architecture missmatch between loaded model and autoencoder definition")
        # self._model = self._vae
        # self._encoder = load_model(path + "/encoder.h5")
        # self._decoder = load_model(path + "/decoder.h5")

    #---------------------------------------------------------------------------------
    def save_model(self, path):
        print("Saving model to {}".format(path))
        #save_model(self._model, path + "/autoencoder.py")
        self.model.save_weights(path + "/autoencoder.h5")
        save_model(self._encoder, path + "/encoder.h5")
        save_model(self._decoder, path + "/decoder.h5")

    #---------------------------------------------------------------------------------
    def encode(self, x, batch_size=32):
        z = self._encoder.predict(x, batch_size=batch_size)
        return z

    #---------------------------------------------------------------------------------
    def decode(self, z, batch_size=32):
        y = self._decoder.predict(x=z, batch_size=batch_size)
        return y

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        """ predict from three dimensional numpy array"""
        # if x[0].ndim == 4:
        #     x = x[:,np.newaxis, ...]
        #     output = self.model.predict_on_batch(x)
        #     return output[0,:,:,:,:]
        # elif x[0].ndim == 5:
        #     return self.model.predict(x, batch_size=batch_size)
        # return np.array([])
        return self.model.predict(x, batch_size=batch_size)

    #---------------------------------------------------------------------------------
    @property
    def encoder_trainable(self):
        return self._trainable_encoder

    #---------------------------------------------------------------------------------
    @encoder_trainable.setter
    def encoder_trainable(self, value):
        self._trainable = value
        make_layers_trainable(self._encoder, value)

class SplitPressureAutoencoder(Autoencoder):
    #---------------------------------------------------------------------------------
    def _build_model(self):
        # ENCODER ################################################################################## 
        input_shape = self.input_shape
        pressure_static_input = Input(shape=input_shape)
        pressure_dynamic_input = Input(shape=input_shape)

        encoder_stages = StagedModel()
        encoder_stages.start([pressure_static_input, pressure_dynamic_input])
        
        encoder_input = Concatenate(axis=4)([pressure_static_input, pressure_dynamic_input])
        # Conv #
        x = Conv3D(self.feature_multiplier * 1, self.surface_kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(encoder_input)
        #x = BatchNormalization(mode=1)(x)
        x = Activation('linear')(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 2, self.kernel_size, strides=(2, 2, 2),padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)
        
        # Conv #
        x = Conv3D(self.feature_multiplier * 4, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 8, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 16, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        x = Conv3D(self.feature_multiplier * 32, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        # Flatten #
        encoder_output_shape = x.shape[1:]#(int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
        self._intermediate_dim = 1024 #int(x.shape[1] * x.shape[2] * x.shape[3])

        # Flatten #
        encoder_output_shape = convert_shape(x.shape[1:]) #(int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
       
        if self.variational_ae:
            self._z_size = encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]#settings.ae.code_layer_size
            x = Flatten()(x)
            x = Dense(units=self._z_size)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.3)(x)
        
        encoder_stages.end(x)

        if self.variational_ae:
            # the following will be added to the autoencoder only and can not be part of the stage models due to some weird keras issue
            z_mean_layer = Dense(units=self._z_size)
            z_log_sigma_layer = Dense(units=self._z_size)
            z_layer = Lambda(_sampling, output_shape=(self._z_size,))

        # DECODER ####################################################################################################
        x = Input(shape=(None, None, None, self.feature_multiplier * 32))
        
        if self.variational_ae:
            x = Input(shape=(self._z_size,))
    
        decoder_stages = StagedModel()
        decoder_stages.start(x)

        if self.variational_ae:
            x = Dense(units=encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3])(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.3)(x)
            x = Reshape(target_shape=encoder_output_shape)(x)
        
        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 16, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 8, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 4, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 2, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)


        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 1, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)        
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(4, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)        
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        
        # Deconv #
        x_static = Conv3DTranspose(1, self.surface_kernel_size, padding='same', kernel_initializer=self.init_func)(x)
        decoder_output_static = Activation('linear')(x_static)

        x_dynamic = Conv3DTranspose(1, self.surface_kernel_size, padding='same', kernel_initializer=self.init_func)(x)
        # x_dynamic = BatchNormalization()(x_dynamic)
        # x_dynamic = LeakyReLU(0.2)(x_dynamic)
        
        # x_dynamic = Conv3DTranspose(1, 1, padding='same', kernel_initializer=self.init_func)(x_dynamic)
        decoder_output_dynamic = Activation('linear')(x_dynamic)
        
        # Stage 1 END #
        decoder_stages.end([decoder_output_static, decoder_output_dynamic])

        self._decoder = decoder_stages.model #Model(input=decoder_input, output = decoder_output)
        self._decoder.name = "Decoder"

        # AUTOENCODER #########################################################################################
        
        # Build staged autoencoder
        self._stages = []
        for stage_num in range(1, len(encoder_stages.stages) + 1):
            stage_input = [Input(shape=input_shape), Input(shape=input_shape)]
            x = encoder_stages[0:stage_num](stage_input)
            stage_output = decoder_stages[6-stage_num:6](x)
            stage_model = Model(inputs=stage_input, outputs=stage_output)
            stage_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            #make_layers_trainable(self._stages[0], False)
            self._stages.append(stage_model)

        # Autoencoder
        ae_input = [Input(shape=input_shape), Input(shape=input_shape)]
        h = encoder_stages[0:6](ae_input)

        if self.variational_ae:
            z_mean = z_mean_layer(h)
            z_log_sigma = z_log_sigma_layer(h)
            z = z_layer([z_mean, z_log_sigma])
        else:
            z = h

        ae_output = decoder_stages[0:6](z)
        self.model = Model(inputs=ae_input, outputs=ae_output)

        if self.variational_ae:
            self.vae_loss = VAELoss(z_mean=z_mean, z_log_var=z_log_sigma, kl_beta = self.kl_beta, loss=MSE(1.0))      
            self.metrics.append(KLDivergence(z_mean=z_mean, z_log_var=z_log_sigma))            
            self.model.compile(loss=self.vae_loss, optimizer=self.optimizer, metrics=self.metrics)#['mae', self.kl_div])
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.name = "Autoencoder"

        # Encoder
        enc_input = [Input(shape=input_shape), Input(shape=input_shape)]
        h = encoder_stages[0:6](enc_input)

        if self.variational_ae:
            z_mean = z_mean_layer(h)
            z_log_sigma = z_log_sigma_layer(h)
            z = z_layer([z_mean, z_log_sigma])
        else:
            z = h

        self._encoder = Model(enc_input, z)
        self._encoder.name = "Encoder"

class DynamicPressureAutoencoder(Autoencoder):
    #---------------------------------------------------------------------------------
    def _build_model(self):
        # ENCODER ################################################################################## 
        input_shape = self.input_shape
        pressure_static_input = Input(shape=input_shape)
        pressure_dynamic_input = Input(shape=input_shape)

        encoder_stages = StagedModel()
        encoder_stages.start([pressure_static_input, pressure_dynamic_input])
        
        encoder_input = Concatenate(axis=4)([pressure_static_input, pressure_dynamic_input])
        # Conv #
        x = Conv3D(self.feature_multiplier * 1, self.surface_kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(encoder_input)
        #x = BatchNormalization(mode=1)(x)
        x = Activation('linear')(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 2, self.kernel_size, strides=(2, 2, 2),padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)
        
        # Conv #
        x = Conv3D(self.feature_multiplier * 4, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 8, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv3D(self.feature_multiplier * 16, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = encoder_stages.stage(x)

        x = Conv3D(self.feature_multiplier * 32, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        # Flatten #
        encoder_output_shape = x.shape[1:]#(int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
        self._intermediate_dim = 1024 #int(x.shape[1] * x.shape[2] * x.shape[3])

        # Flatten #
        encoder_output_shape = convert_shape(x.shape[1:]) #(int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
       
        if self.variational_ae:
            self._z_size = encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]#settings.ae.code_layer_size
            x = Flatten()(x)
            x = Dense(units=self._z_size)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.3)(x)
        
        encoder_stages.end(x)

        if self.variational_ae:
            # the following will be added to the autoencoder only and can not be part of the stage models due to some weird keras issue
            z_mean_layer = Dense(units=self._z_size)
            z_log_sigma_layer = Dense(units=self._z_size)
            z_layer = Lambda(_sampling, output_shape=(self._z_size,))

        # DECODER ####################################################################################################
        x = Input(shape=(None, None, None, self.feature_multiplier * 32))
        
        if self.variational_ae:
            x = Input(shape=(self._z_size,))
    
        decoder_stages = StagedModel()
        decoder_stages.start(x)

        if self.variational_ae:
            x = Dense(units=encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3])(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.3)(x)
            x = Reshape(target_shape=encoder_output_shape)(x)
        
        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 16, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 8, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 4, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 2, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.2)(x)

        x = decoder_stages.stage(x)


        # Deconv #
        x = Conv3DTranspose(self.feature_multiplier * 1, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)        
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv3DTranspose(4, self.kernel_size, strides=(2, 2, 2), padding='same', kernel_initializer=self.init_func)(x)        
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        
        # Deconv #

        x_dynamic = Conv3DTranspose(1, self.surface_kernel_size, padding='same', kernel_initializer=self.init_func)(x)
        # x_dynamic = BatchNormalization()(x_dynamic)
        # x_dynamic = LeakyReLU(0.2)(x_dynamic)
        
        # x_dynamic = Conv3DTranspose(1, 1, padding='same', kernel_initializer=self.init_func)(x_dynamic)
        decoder_output_dynamic = Activation('linear')(x_dynamic)
        
        # Stage 1 END #
        decoder_stages.end(decoder_output_dynamic)

        self._decoder = decoder_stages.model #Model(input=decoder_input, output = decoder_output)
        self._decoder.name = "Decoder"

        # AUTOENCODER #########################################################################################
        
        # Build staged autoencoder
        self._stages = []
        for stage_num in range(1, len(encoder_stages.stages) + 1):
            stage_input = [Input(shape=input_shape), Input(shape=input_shape)]
            x = encoder_stages[0:stage_num](stage_input)
            stage_output = decoder_stages[6-stage_num:6](x)
            stage_model = Model(inputs=stage_input, outputs=stage_output)
            stage_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            #make_layers_trainable(self._stages[0], False)
            self._stages.append(stage_model)

        # Autoencoder
        ae_input = [Input(shape=input_shape), Input(shape=input_shape)]
        h = encoder_stages[0:6](ae_input)

        if self.variational_ae:
            z_mean = z_mean_layer(h)
            z_log_sigma = z_log_sigma_layer(h)
            z = z_layer([z_mean, z_log_sigma])
        else:
            z = h

        ae_output = decoder_stages[0:6](z)
        self.model = Model(inputs=ae_input, outputs=ae_output)

        if self.variational_ae:
            self.vae_loss = VAELoss(z_mean=z_mean, z_log_var=z_log_sigma, kl_beta = self.kl_beta, loss=MSE(1.0))      
            self.metrics.append(KLDivergence(z_mean=z_mean, z_log_var=z_log_sigma))            
            self.model.compile(loss=self.vae_loss, optimizer=self.optimizer, metrics=self.metrics)#['mae', self.kl_div])
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.name = "Autoencoder"

        # Encoder
        enc_input = [Input(shape=input_shape), Input(shape=input_shape)]
        h = encoder_stages[0:6](enc_input)

        if self.variational_ae:
            z_mean = z_mean_layer(h)
            z_log_sigma = z_log_sigma_layer(h)
            z = z_layer([z_mean, z_log_sigma])
        else:
            z = h

        self._encoder = Model(enc_input, z)
        self._encoder.name = "Encoder"

        #---------------------------------------------------------------------------------
    def _pretrain(self, dataset, epochs = 10, batch_size=128, plot_callback=None, augment=False, noise=False):
        """
        Train the autoencoder in stages of increasing size 
        :param epochs: Number of epochs of training
        :param x: The dataset fed as input to the autoencoder
        :param y: The real dataset, by which the autoencoder will be measured
        """
        hist = None
        callbacks = None
        if plot_callback is not None:
            loss_hist = LossHistory(plot_callback)
            callbacks = [loss_hist]

        if epochs == 0:
            return None
        try:
            print("\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("Autoencoder pretraining")
            train_generator = dataset.train.generator(batch_size=batch_size, outputs=["pressure_dynamic"], augment=augment, noise=noise)
            train_steps_per_epoch = dataset.train.steps_per_epoch(batch_size=batch_size, augment=augment)
            val_generator = dataset.val.generator(batch_size=batch_size, outputs=["pressure_dynamic"], augment=augment, noise=noise)
            val_steps = dataset.val.steps_per_epoch(batch_size=batch_size, augment=augment)
            # Train the autoencoder stages
            for index, stage in enumerate(self._stages):
                print("\n--------------------------------------------------------------------------------")
                print("Stage {}\n".format(index + 1))
                
                hist = stage.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    epochs=epochs,
                    max_queue_size=100,
                    callbacks=callbacks)
                    
        except KeyboardInterrupt:
            print("\n Interrupted by user")
        
        #self.model = stage
        return hist

    #---------------------------------------------------------------------------------
    def _train_full_model(self, dataset, epochs = 10, batch_size=128, plot_callback=None, augment=False, noise=False):
        hist = None
        callbacks = None
        if plot_callback is not None:
            loss_hist = LossHistory(plot_callback)
            callbacks = [loss_hist]

        if epochs == 0:
            return None
        try:
            print("\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("Autoencoder training")
            train_generator = dataset.train.generator(batch_size=batch_size, outputs=["pressure_dynamic"], augment=augment, noise=noise)
            train_steps_per_epoch = dataset.train.steps_per_epoch(batch_size=batch_size, augment=augment)
            val_generator = dataset.val.generator(batch_size=batch_size, outputs=["pressure_dynamic"], augment=augment, noise=noise)
            val_steps = dataset.val.steps_per_epoch(batch_size=batch_size, augment=augment)

            hist = self.model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=val_generator,
                validation_steps=val_steps,
                epochs=epochs,
                max_queue_size=100,
                callbacks=callbacks)
                
        except KeyboardInterrupt:
            print("\n Interrupted by user")
    
        return hist





# 2D --------------------------------------------------------------------------------------------------------------------------------------------------
#=====================================================================================
class Autoencoder2D(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, settings=None, **kwargs):
        self.init_func = "glorot_normal"
        self.feature_multiplier = 8
        self.surface_kernel_size = 4
        self.kernel_size = 2
        self.adam_epsilon = None #1e-8 # 1e-3
        self.adam_learning_rate = 0.001

        self.adam_lr_decay = 0.0005 #1e-5
        self.pretrain_epochs = 1 #3
        self.dropout = kwargs.get("dropout", 0.0)
        self.input_shape = kwargs.get("input_shape", (64, 64, 1))
        # self.loss = kwargs.get("loss", "mse")
        # self.metrics = ["mae"]
        # if not isinstance(self.loss, str):
        #     self.metrics = ["mse", "mae"]
        self.set_loss(loss=kwargs.get("loss", "mse"))
        self.l1_reg = kwargs.get("l1_reg", 0.0)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.variational_ae = False
        self.kl_beta = 1e-5
        self.tensorflow_seed = kwargs.get("tensorflow_seed", 4)
        self.model = None
        tf.set_random_seed(self.tensorflow_seed)

    #---------------------------------------------------------------------------------
    def set_loss(self, loss):
        self.loss = loss
        self.metrics = ["mae"]
        if not isinstance(self.loss, str):
            self.metrics = ["mse", "mae"]
            self.metrics.append(Loss(
                loss_type=LossType.weighted_tanhmse_mse,
                loss_ratio=1.0,
                data_input_scale=1.0))

    #---------------------------------------------------------------------------------
    def _init_optimizer(self, epochs=1):
        self.optimizer = Adam(lr=self.adam_learning_rate, epsilon=self.adam_epsilon, decay=self.adam_lr_decay)
        self.kernel_regularizer = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        return self.optimizer

    #---------------------------------------------------------------------------------
    def _build_model(self):
        # ENCODER ################################################################################## 
        input_shape = self.input_shape
        pressure_input = Input(shape=input_shape)

        encoder_stages = StagedModel()
        encoder_stages.start(pressure_input)
        
        encoder_input = pressure_input
        x = encoder_input

        # Conv #
        x = Conv3D(self.feature_multiplier * 1, self.surface_kernel_size, strides=(2, 2, 2), padding='same',
                   kernel_initializer=self.init_func)(encoder_input)

        x = Conv2D(self.feature_multiplier * 1, self.surface_kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2D(self.feature_multiplier * 1, self.surface_kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2D(self.feature_multiplier * 1, self.surface_kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        #x = BatchNormalization(mode=1)(x)
        #x = Activation('linear')(x)
        x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv2D(self.feature_multiplier * 2, self.kernel_size, strides=(1, 1),padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2D(self.feature_multiplier * 2, self.kernel_size, strides=(2, 2),padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        #x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = encoder_stages.stage(x)
        
        # Conv #
        x = Conv2D(self.feature_multiplier * 4, self.kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2D(self.feature_multiplier * 4, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv2D(self.feature_multiplier * 8, self.kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2D(self.feature_multiplier * 8, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        #x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = encoder_stages.stage(x)

        # Conv #
        x = Conv2D(self.feature_multiplier * 16, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = encoder_stages.stage(x)

        x = Conv2D(self.feature_multiplier * 32, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        # Flatten #
        print("Enc Out Shape: {}".format(x.shape[1:]))
        encoder_output_shape = convert_shape(x.shape[1:]) # (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
       
        if self.variational_ae:
            self._z_size = encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2]
            x = Flatten()(x)
            x = Dense(units=self._z_size)(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = BatchNormalization()(x)
        
        # Stage 4 END #
        encoder_stages.end(x)

        if self.variational_ae:
            # the following will be added to the autoencoder only and can not be part of the stage models due to some weird keras issue
            z_mean_layer = Dense(units=self._z_size)
            z_log_sigma_layer = Dense(units=self._z_size)
            z_layer = Lambda(_sampling, output_shape=(self._z_size,))

        # DECODER ####################################################################################################
        x = Input(shape=(None, None, self.feature_multiplier * 32))

        if self.variational_ae:
            decoder_input = Input(shape=(self._z_size,))
            x = decoder_input
    
        # Stage 4 BEGIN #
        decoder_stages = StagedModel()
        decoder_stages.start(x)

        if self.variational_ae:
            x = Dense(units=encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2])(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = BatchNormalization()(x)
            x = Reshape(target_shape=encoder_output_shape)(x)

        # Deconv #
        x = Conv2DTranspose(self.feature_multiplier * 16, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv2DTranspose(self.feature_multiplier * 8, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        #x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = Conv2DTranspose(self.feature_multiplier * 8, self.kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv2DTranspose(self.feature_multiplier * 4, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        #x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = Conv2DTranspose(self.feature_multiplier * 4, self.kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv2DTranspose(self.feature_multiplier * 2, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        #x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = Conv2DTranspose(self.feature_multiplier * 2, self.kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = decoder_stages.stage(x)

        # Deconv #
        x = Conv2DTranspose(self.feature_multiplier * 1, self.kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)        
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(self.feature_multiplier * 1, self.kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x) if self.dropout > 0.0 else x

        x = decoder_stages.stage(x)
        
        # Deconv #
        x = Conv2DTranspose(input_shape[-1], self.surface_kernel_size, strides=(2, 2), padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer)(x)
        #x = Activation('tanh')(x)

        #x = Conv2DTranspose(input_shape[-1], self.surface_kernel_size, strides=(1, 1), padding='same', kernel_initializer=self.init_func)(x)

        decoder_output = Activation('linear')(x)

        # Stage 1 END #
        decoder_stages.end(decoder_output)

        self._decoder = decoder_stages.model #Model(input=decoder_input, output = decoder_output)
        self._decoder.name = "Decoder"

        # AUTOENCODER #########################################################################################

        max_enc_stages = len(encoder_stages.stages)
        max_dec_stages = len(decoder_stages.stages)

        # Build staged autoencoder
        self._stages = []
        for stage_num in range(1, max_enc_stages + 1):
            stage_input = Input(shape=input_shape)
            x = encoder_stages[0:stage_num](stage_input)
            stage_output = decoder_stages[max_dec_stages-stage_num:max_dec_stages](x)
            stage_model = Model(inputs=stage_input, outputs=stage_output)
            stage_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            #make_layers_trainable(self._stages[0], False)
            self._stages.append(stage_model)

        # Autoencoder
        ae_input = Input(shape=input_shape)
        h = encoder_stages[0:max_enc_stages](ae_input)

        if self.variational_ae:
            z_mean = z_mean_layer(h)
            z_log_sigma = z_log_sigma_layer(h)
            z = z_layer([z_mean, z_log_sigma])
        else:
            z = h

        ae_output = decoder_stages[0:max_dec_stages](z)
        self.model = Model(inputs=ae_input, outputs=ae_output)

        if self.variational_ae:
            self.vae_loss = VAELoss(z_mean=z_mean, z_log_var=z_log_sigma, kl_beta = self.kl_beta, loss=MSE(1.0))
            self.metrics.append(KLDivergence(z_mean=z_mean, z_log_var=z_log_sigma))
            self.model.compile(loss=self.vae_loss, optimizer=self.optimizer, metrics=self.metrics)
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            
        self.model.name = "Autoencoder"

        # Encoder
        enc_input = Input(shape=input_shape)
        max_enc_stages = len(encoder_stages.stages)
        h = encoder_stages[0:max_enc_stages](enc_input)

        if self.variational_ae:
            z_mean = z_mean_layer(h)
            z_log_sigma = z_log_sigma_layer(h)
            z = z_layer([z_mean, z_log_sigma])
        else:
            z = h

        self._encoder = Model(enc_input, z)
        self._encoder.name = "Encoder"

    #---------------------------------------------------------------------------------
    def _compile_model(self):
        pass

    #---------------------------------------------------------------------------------
    def _train(self, epochs, **kwargs):
        # get the relevant arguments
        dataset = kwargs.get("dataset", None)
        assert dataset is not None, ("You must provide a dataset argument to autoencoder train")
        batch_size = kwargs.get("batch_size", 32) 
        augment = kwargs.get("augment", False)
        
        # pretrain if enabled
        if not kwargs.get("disable_pretrain", False):
            self._pretrain(dataset=dataset, epochs=self.pretrain_epochs, batch_size=batch_size, augment=augment)
        
        # do the actual training
        return self._train_full_model(dataset=dataset, epochs=epochs, batch_size=batch_size, augment=augment, plot_evaluation_callback=kwargs.get("plot_evaluation_callback", None))

    #---------------------------------------------------------------------------------
    def _pretrain(self, dataset, epochs = 10, batch_size=128, plot_callback=None, augment=False, noise=False):
        """ 
        Train the autoencoder in stages of increasing size 
        :param epochs: Number of epochs of training
        :param x: The dataset fed as input to the autoencoder
        :param y: The real dataset, by which the autoencoder will be measured
        """
        hist = None
        callbacks = None
        if plot_callback is not None:
            loss_hist = LossHistory(plot_callback)
            callbacks = [loss_hist]

        if epochs == 0:
            return None
        try:
            print("\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("Autoencoder pretraining")
            train_generator = dataset.train.generator(batch_size=batch_size, augment=augment, noise=noise)
            train_steps_per_epoch = dataset.train.steps_per_epoch(batch_size=batch_size, augment=augment)
            val_generator = dataset.val.generator(batch_size=batch_size, augment=augment, noise=noise)
            val_steps = dataset.val.steps_per_epoch(batch_size=batch_size, augment=augment)
            # Train the autoencoder stages
            for index, stage in enumerate(self._stages):
                print("\n--------------------------------------------------------------------------------")
                print("Stage {}\n".format(index + 1))
                hist = stage.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    epochs=epochs,
                    max_queue_size=100,
                    callbacks=callbacks)
                    
        except KeyboardInterrupt:
            print("\n Interrupted by user")
        
        #self.model = stage
        return hist

    #---------------------------------------------------------------------------------
    def encoder_output_shape(self, input_shape=None):
        if not None in self.input_shape:
            input_shape = self.input_shape
        assert input_shape is not None, ("You must provide an input shape for autoencoders with variable input sizes")
        dummy_input = np.expand_dims(np.zeros(input_shape), axis = 0)
        shape = self.encode(dummy_input, 1).shape[1:]
        return shape

    #---------------------------------------------------------------------------------
    def _train_full_model(self, dataset, epochs = 10, batch_size=128, plot_loss_hist_callback=None, augment=False, noise=False, plot_evaluation_callback=None):
        hist = None
        callbacks = []
        if plot_loss_hist_callback is not None:
            loss_hist = LossHistory(plot_loss_hist_callback)
            callbacks.append(loss_hist)
        if plot_evaluation_callback is not None:
            callbacks.append(plot_evaluation_callback)

        if epochs == 0:
            return None
        try:
            print("\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("Autoencoder training")
            train_generator = dataset.train.generator(batch_size=batch_size, augment=augment, noise=noise)
            train_steps_per_epoch = dataset.train.steps_per_epoch(batch_size=batch_size, augment=augment)
            val_generator = dataset.val.generator(batch_size=batch_size, augment=augment, noise=noise)
            val_steps = dataset.val.steps_per_epoch(batch_size=batch_size, augment=augment)

            hist = self.model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=val_generator,
                validation_steps=val_steps,
                epochs=epochs,
                max_queue_size=100,
                callbacks=callbacks)
                
        except KeyboardInterrupt:
            print("\n Interrupted by user")
    
        return hist

    #---------------------------------------------------------------------------------
    def print_summary(self):
        print("Autoencoder")
        self.model.summary()
        print("Encoder")
        self._encoder.summary()
        print("Decoder")
        self._decoder.summary()

    #---------------------------------------------------------------------------------
    def load_model(self, path):
        print("Loading model from {}".format(path))
        #self._vae = load_model(path)
        if self.model is None:
            self._build_model()
        self.model.load_weights(path, by_name=True)
        # if self._vae.to_json() != self._model.to_json():
        #     print("[WARNING] Architecture missmatch between loaded model and autoencoder definition")
        # self._model = self._vae
        # self._encoder = load_model(path + "/encoder.h5")
        # self._decoder = load_model(path + "/decoder.h5")

    #---------------------------------------------------------------------------------
    def save_model(self, path):
        print("Saving model to {}".format(path))
        #save_model(self._model, path + "/autoencoder.py")
        self.model.save_weights(path + "/autoencoder.h5")
        save_model(self._encoder, path + "/encoder.h5")
        save_model(self._decoder, path + "/decoder.h5")

    #---------------------------------------------------------------------------------
    def encode(self, x, batch_size=32):
        z = self._encoder.predict(x, batch_size=batch_size)
        return z

    #---------------------------------------------------------------------------------
    def decode(self, z, batch_size=32):
        y = self._decoder.predict(x=z, batch_size=batch_size)
        return y

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        """ predict from three dimensional numpy array"""
        return self.model.predict(x, batch_size=batch_size)

    #---------------------------------------------------------------------------------
    @property
    def encoder_trainable(self):
        return self._trainable_encoder

    #---------------------------------------------------------------------------------
    @encoder_trainable.setter
    def encoder_trainable(self, value):
        self._trainable = value
        make_layers_trainable(self._encoder, value)


class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200


        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = args.lr


    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):



            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [len(self.test_x), self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [len(self.test_y), self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)
        self.test_logits = self.network(self.test_inptus, is_training=False, reuse=True)

        self.train_loss, self.train_accuracy = classification_loss(logit=self.train_logits, label=self.train_labels)
        self.test_loss, self.test_accuracy = classification_loss(logit=self.test_logits, label=self.test_labels)

        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss


        """ Training """
        self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus : self.test_x,
                    self.test_labels : self.test_y
                }


                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy = self.sess.run(
                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: self.test_x,
            self.test_labels: self.test_y
        }


        test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))