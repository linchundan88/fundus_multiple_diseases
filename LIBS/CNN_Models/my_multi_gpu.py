from keras import Model
from keras.utils import multi_gpu_model

'''
Keras2.2.4 fix this bug.

为了解决multi_gpu checkpoint 的保存问题, 保存的时候用原来的模型保存
https://keras.io/LIBS/#multi_gpu_model

multi_gpu_model
keras.LIBS.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False)

Replicates a model on different GPUs.
Specifically, this function implements single-machine multi-GPU data1 parallelism. It works in the following way:

Divide the model's input(s) into multiple sub-batches.
Apply a model copy on each sub-batch. Every model copy is executed on a dedicated GPU.
Concatenate the results (on CPU) into one big batch.

'''


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
