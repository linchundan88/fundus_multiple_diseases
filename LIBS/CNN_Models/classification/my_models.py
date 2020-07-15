from keras.applications import InceptionV3, Xception, InceptionResNetV2, densenet, MobileNetV2

from keras_applications import nasnet
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

def add_multilabels_top(base_model,NUM_CLASSES):

    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def get_models(model_name, NUM_CLASSES):

    #region Normal Inceptionv3,Xception,InceptionResnetV2
    if model_name == 'InceptionV3':
        model = InceptionV3(include_top=True, input_shape=(299, 299, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32
        # BATCH_SIZE_TRAIN = 64  # 增加GPU之后
    elif model_name == 'InceptionV4':
        from LIBS.CNN_Models.classification import inception_v4
        model = inception_v4.create_inception_v4(nb_classes=NUM_CLASSES, load_weights=False)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 16
    elif model_name == 'Xception':
        model = Xception(include_top=True, input_shape=(299, 299, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 28  # 增加GPU之后
    elif model_name == 'InceptionResNetV2':
        # following keras implemantation don't use dropout
        # 标准的不用dropout，收敛快了许多, train准确率提高
        model = InceptionResNetV2(include_top=True, input_shape=(299, 299, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32  # 增加GPU之后
    elif model_name == 'MobileNetV2':   #Total params: 2,261,827
        # model = MobileNetV2(include_top=True, input_shape=(256, 256, 3), weights=None, classes=NUM_CLASSES)
        model = MobileNetV2(include_top=True, input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 64
    elif model_name == 'My_Xception':
        model = my_xception.Xception(classes=NUM_CLASSES)

        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32  # 增加GPU之后
    elif model_name == 'DRN_A18': #params:20,631,682
        from LIBS.CNN_Models.classification import my_DRN_A
        model = my_DRN_A.DRN_A_Builder.build_DRN_A_18(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)

        IMAGE_SIZE = 256
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'DRN_A34':
        from LIBS.CNN_Models.classification import my_DRN_A_1
        # model = my_DRN_A.DRN_A_Builder.build_DRN_A_34(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)
        model = my_DRN_A_1.DRN_A_Builder.build_DRN_A_34(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)

        IMAGE_SIZE = 256
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'DRN_A50':
        from LIBS.CNN_Models.classification import my_DRN_A
        model = my_DRN_A.DRN_A_Builder.build_DRN_A_50(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)

        IMAGE_SIZE = 256
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'DRN_A101':
        from LIBS.CNN_Models.classification import my_DRN_A
        model = my_DRN_A.DRN_A_Builder.build_DRN_A_101(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)

        IMAGE_SIZE = 256
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'DRN_C26': #params:20,631,682
        from LIBS.CNN_Models.classification import my_DRN_C
        model = my_DRN_C.DRN_C_Builder.build_DRN_C_26(input_shape=(224, 224, 3), num_classes=NUM_CLASSES)

        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'DRN_C42':   # params:30,750,978
        from LIBS.CNN_Models.classification import my_DRN_C
        model = my_DRN_C.DRN_C_Builder.build_DRN_C_42(input_shape=(224, 224, 3), num_classes=NUM_CLASSES)

        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'DPN92':   #DPN92,DPN98,DPN107,DPN137
        from LIBS.CNN_Models.classification import my_dpn
        model = my_dpn.DPN92(input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 16  # GTX1080 ti 32 exhausted
    elif model_name == 'DPN98':
        from LIBS.CNN_Models.classification import my_dpn
        model = my_dpn.DPN92(input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 16  # GTX1080 ti 32 exhausted
    elif model_name == 'DPN107':
        from LIBS.CNN_Models.classification import my_dpn
        model = my_dpn.DPN107(input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 16  # GTX1080 ti 32 exhausted
    elif model_name == 'DPN137':
        from LIBS.CNN_Models.classification import my_dpn
        model = my_dpn.DPN137(input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 16  # GTX1080 ti 32 exhausted
    elif model_name =='DenseNet121':   #121,169,201
        model =densenet.DenseNet121(input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 32
    elif model_name =='DenseNet169':
        model =densenet.DenseNet169(input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 32
    elif model_name =='DenseNet201':
        model =densenet.DenseNet201(input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 16
    elif model_name =='NasnetMobile':
        model =nasnet.NASNetMobile(input_shape=(224, 224, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'my_Mnasnet':
        from LIBS.CNN_Models.classification import my_Mnasnet
        model = my_Mnasnet.MnasNet(input_shape=(224, 224, 3), classes=NUM_CLASSES)
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 64
    elif model_name == "Mnasnet":
        from LIBS.CNN_Models.classification import my_Mnasnet
        model = my_Mnasnet.MnasNet(classes=NUM_CLASSES, input_shape=(224, 224, 3))
        IMAGE_SIZE = 224
        BATCH_SIZE_TRAIN = 64
    elif model_name == 'NasnetMedium':     #Trainable params: 22,695,624
        model = nasnet.NASNet(input_shape=(299, 299, 3),
            penultimate_filters=1920, num_blocks=7, stem_block_filters=80,  # Large:96, Mobile:32
            classes=NUM_CLASSES,  default_size=299)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 16
    elif model_name == 'NasnetLarge':
        model = nasnet.NASNetLarge(input_shape=(331, 331, 3), weights=None, classes=NUM_CLASSES)
        IMAGE_SIZE = 331
        BATCH_SIZE_TRAIN = 8
    #endregion

    #region All kinds of  ResNet, ResNeXt
    # 7*32=224, 256, 288, 320, 352, 384
    elif model_name == 'ResNet50':
        from LIBS.CNN_Models.classification import my_resnet
        model = my_resnet.ResnetBuilder.build_resnet_50((256, 256, 3), NUM_CLASSES)
        IMAGE_SIZE = 256
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'ResNet50_288':
        from LIBS.CNN_Models.classification import my_resnet
        model = my_resnet.ResnetBuilder.build_resnet_50((288, 288, 3), NUM_CLASSES)
        IMAGE_SIZE = 288
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'ResNet101':
        from LIBS.CNN_Models.classification import my_resnet
        model = my_resnet.ResnetBuilder.build_resnet_101((256, 256, 3), NUM_CLASSES)
        IMAGE_SIZE = 256
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'ResNet448':
        from LIBS.CNN_Models.classification import my_resnet
        # model = my_resnet.ResnetBuilder.build_resnet_mymodel_34_64_5((448, 448, 3), NUM_CLASSES)
        model = my_resnet.ResnetBuilder.build_resnet_448_1((448, 448, 3), NUM_CLASSES)
        IMAGE_SIZE = 448
        BATCH_SIZE_TRAIN = 32  # 增加GPU之后
    elif model_name == 'ResNext448':
        from LIBS.CNN_Models.classification import resNeXt
        model = resNeXt.my_ResNext(input_shape=(448, 448, 3), classes=NUM_CLASSES)
        IMAGE_SIZE = 448
        BATCH_SIZE_TRAIN = 16  # 增加GPU之后

    #endregion


    #region multi label (Inceptionv3, Xception, InceptionResnetV2)
    elif model_name == 'Multi_label_InceptionV3':
        base_model = InceptionV3(include_top=False, weights=None)
        model = add_multilabels_top(base_model, NUM_CLASSES)

        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'Multi_label_SE_InceptionV3':
        from LIBS.CNN_Models.classification import my_se_inception_v3
        base_model = my_se_inception_v3.SE_InceptionV3((299, 299, 3), include_top=False)
        model = add_multilabels_top(base_model, NUM_CLASSES)

        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'Multi_label_Xception':
        base_model = Xception(include_top=False, weights=None)
        model = add_multilabels_top(base_model, NUM_CLASSES)

        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32  # 增加GPU之后
    elif model_name == 'Multi_label_my_Xception':
        from LIBS.CNN_Models.classification import my_xception
        model = my_xception.Xception(classes=NUM_CLASSES, multi_labels=True)

        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32  # 增加GPU之后

    elif model_name == 'Multi_label_InceptionResNetV2':
        base_model = InceptionResNetV2(include_top=False, weights=None)
        model = add_multilabels_top(base_model, NUM_CLASSES)

        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32

    elif model_name == 'Multi_label_DRN_A_Xception':
        from LIBS.CNN_Models import DRN_A_Xception_Builder
        base_model = DRN_A_Xception_Builder.build_DRN_A_xception(input_shape=(288, 288, 3),
                                                                 num_classes=29, include_top=False)

        model = add_multilabels_top(base_model, NUM_CLASSES)
        IMAGE_SIZE = 288
        BATCH_SIZE_TRAIN = 32

    elif model_name == 'Multi_NasnetMedium':
        base_model = nasnet.NASNet(input_shape=(299, 299, 3),
                              penultimate_filters=1920, num_blocks=7, stem_block_filters=80,  # Large:96, Mobile:32
                              classes=NUM_CLASSES, default_size=299, include_top=False)
        model = add_multilabels_top(base_model, NUM_CLASSES)

        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 16
    elif model_name == 'Multi_DRN_A18':
        from LIBS.CNN_Models.classification import my_DRN_A
        #original number of parameters, resnet34 :21.8M, resnet50:25.6M
        base_model = my_DRN_A.DRN_A_Builder.build_DRN_A_18(input_shape=(288, 288, 3), num_classes=NUM_CLASSES, include_top=False)

        model = add_multilabels_top(base_model, NUM_CLASSES)
        IMAGE_SIZE = 288
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'Multi_DRN_A34':
        from LIBS.CNN_Models.classification import my_DRN_A
        base_model = my_DRN_A.DRN_A_Builder.build_DRN_A_34(input_shape=(288, 288, 3), num_classes=NUM_CLASSES, include_top=False)

        model = add_multilabels_top(base_model, NUM_CLASSES)
        IMAGE_SIZE = 288
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'Multi_DRN_A50':
        from LIBS.CNN_Models.classification import my_DRN_A
        base_model = my_DRN_A.DRN_A_Builder.build_DRN_A_50(input_shape=(288, 288, 3), num_classes=NUM_CLASSES, include_top=False)

        model = add_multilabels_top(base_model, NUM_CLASSES)
        IMAGE_SIZE = 288
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'Multi_DRN_C26': #params:20,631,682
        from LIBS.CNN_Models.classification import my_DRN_C
        base_model = my_DRN_C.DRN_C_Builder.build_DRN_C_26(input_shape=(288, 288, 3), num_classes=NUM_CLASSES, include_top=False)

        model = add_multilabels_top(base_model, NUM_CLASSES)
        IMAGE_SIZE = 288
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'Multi_DRN_C42':   # params:30,750,978
        from LIBS.CNN_Models.classification import my_DRN_C
        base_model = my_DRN_C.DRN_C_Builder.build_DRN_C_42(input_shape=(288, 288, 3), num_classes=NUM_CLASSES, include_top=False)

        model = add_multilabels_top(base_model, NUM_CLASSES)
        IMAGE_SIZE = 288
        BATCH_SIZE_TRAIN = 32

    elif model_name == 'Multi_DRN_C58':
        from LIBS.CNN_Models.classification import my_DRN_C
        base_model = my_DRN_C.DRN_C_Builder.build_DRN_C_58(input_shape=(288, 288, 3), num_classes=NUM_CLASSES, include_top=False)

        model = add_multilabels_top(base_model, NUM_CLASSES)
        IMAGE_SIZE = 288
        BATCH_SIZE_TRAIN = 32
    # endregion


    #region SE Net(Se_InceptionV3, Se_InceptionResNetV2等)
    elif model_name == 'Se_InceptionV3':
        from LIBS.CNN_Models.classification import my_se_inception_v3
        model = my_se_inception_v3.SE_InceptionV3((299, 299, 3), classes=NUM_CLASSES)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 48

    elif model_name == 'Se_InceptionResNetV2':
        from LIBS.CNN_Models.classification import my_se_inception_resnet_v2
        model = my_se_inception_resnet_v2.SE_InceptionResNetV2((299, 299, 3), classes=NUM_CLASSES)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32
    elif model_name == 'Se_Resnext':
        from LIBS.CNN_Models import my_se_resnext
        model = my_se_resnext.SEResNextImageNet((299, 299, 3), classes=NUM_CLASSES)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 24
    elif model_name == 'Se_Resnet50':
        from LIBS.CNN_Models import my_se_resnet
        model = my_se_resnet.SEResNet50((299, 299, 3), classes=NUM_CLASSES)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 32
    #endregion


    '''other models
    #region multi label SE_NET
    elif model_name == 'Multi_label_Se_InceptionV3':
        from CNN_Models import my_se_inception_v3
        base_model = my_se_inception_v3.SE_InceptionV3((299, 299, 3), include_top=False,
                                                       classes=29, weights=None)

        x = base_model.output
        from keras.layers import GlobalAveragePooling2D, Dense
        from keras.models import Model
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 48
    elif model_name == 'Multi_label_Se_InceptionResNetV2':
        from CNN_Models import my_se_inception_resnet_v2
        base_model = my_se_inception_resnet_v2.SE_InceptionResNetV2((299, 299, 3), include_top=False,
                                                        classes=29, weights=None)

        x = base_model.output
        from keras.layers import GlobalAveragePooling2D, Dense
        from keras.models import Model
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        IMAGE_SIZE = 299
        BATCH_SIZE_TRAIN = 24
        BATCH_SIZE_TRAIN = 32  # 增加GPU之后

    #endregion

    '''

    return model, IMAGE_SIZE, BATCH_SIZE_TRAIN