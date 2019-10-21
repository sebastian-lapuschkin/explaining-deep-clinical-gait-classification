#SVM models
from .svm import SvmLinearL2C1e0
from .svm import SvmLinearL2C1em1
from .svm import SvmLinearL2C5em2
from .svm import SvmLinearL2C1em2
from .svm import SvmLinearL2C5em3
from .svm import SvmLinearL2C1em3
from .svm import SvmLinearL2C1ep1

#MLP models
from .mlp import MlpLinear

from .mlp import Mlp2Layer64Unit
from .mlp import Mlp2Layer128Unit
from .mlp import Mlp2Layer256Unit
from .mlp import Mlp2Layer512Unit
from .mlp import Mlp2Layer768Unit

from .mlp import Mlp3Layer64Unit
from .mlp import Mlp3Layer128Unit
from .mlp import Mlp3Layer256Unit
from .mlp import Mlp3Layer512Unit
from .mlp import Mlp3Layer768Unit

from .mlp import Mlp3Layer64UnitLongerTraining
from .mlp import Mlp3Layer128UnitLongerTraining
from .mlp import Mlp3Layer256UnitLongerTraining
from .mlp import Mlp3Layer512UnitLongerTraining
from .mlp import Mlp3Layer768UnitLongerTraining

from .mlp import Mlp3Layer64UnitLongerTrainingDecreaseBatchSize
from .mlp import Mlp3Layer128UnitLongerTrainingDecreaseBatchSize
from .mlp import Mlp3Layer256UnitLongerTrainingDecreaseBatchSize
from .mlp import Mlp3Layer512UnitLongerTrainingDecreaseBatchSize
from .mlp import Mlp3Layer768UnitLongerTrainingDecreaseBatchSize

#CNN-A models (reading all input channels at once, moving over time axis)
from .cnn import CnnA3
from .cnn import CnnA6
from .cnn import CnnAshort

#CNN-C models (1d- and 2d-convolutional models)
from .cnn import CnnC3
from .cnn import CnnC6
from .cnn import CnnC3_3

from .cnn import Cnn1DC3
from .cnn import Cnn1DC6
from .cnn import Cnn1DC8

from .cnn import Cnn1DC3_Tanh
from .cnn import Cnn1DC8_Tanh

from .cnn import Cnn1DC3_C
from .cnn import Cnn1DC3_CTanh

from .cnn import Cnn1DC8_C
from .cnn import Cnn1DC8_CTanh

from .cnn import Cnn1DC3_D
from .cnn import Cnn1DC3_DTanh

from .cnn import Cnn1DC8_D
from .cnn import Cnn1DC8_DTanh



#all below classes need to be registered below and vice versa in order to use the creator-pattern
def get_architecture(name):
    architectures = {
        'SvmLinearL2C1e0'.lower():SvmLinearL2C1e0,
        'SvmLinearL2C1em1'.lower():SvmLinearL2C1em1,
        'SvmLinearL2C5em2'.lower():SvmLinearL2C5em2,
        'SvmLinearL2C1em2'.lower():SvmLinearL2C1em2,
        'SvmLinearL2C5em3'.lower():SvmLinearL2C5em3,
        'SvmLinearL2C1em3'.lower():SvmLinearL2C1em3,
        'SvmLinearL2C1ep1'.lower():SvmLinearL2C1ep1,

        'MlpLinear'.lower():MlpLinear,

        "Mlp2Layer64Unit".lower():Mlp2Layer64Unit,
        "Mlp2Layer128Unit".lower():Mlp2Layer128Unit,
        "Mlp2Layer256Unit".lower():Mlp2Layer256Unit,
        "Mlp2Layer512Unit".lower():Mlp2Layer512Unit,
        "Mlp2Layer768Unit".lower():Mlp2Layer768Unit,

        "Mlp3Layer64Unit".lower():Mlp3Layer64Unit,
        "Mlp3Layer128Unit".lower():Mlp3Layer128Unit,
        "Mlp3Layer256Unit".lower():Mlp3Layer256Unit,
        "Mlp3Layer512Unit".lower():Mlp3Layer512Unit,
        "Mlp3Layer768Unit".lower():Mlp3Layer768Unit,

        "Mlp3Layer64UnitLongerTraining".lower():Mlp3Layer64UnitLongerTraining,
        "Mlp3Layer128UnitLongerTraining".lower():Mlp3Layer128UnitLongerTraining,
        "Mlp3Layer256UnitLongerTraining".lower():Mlp3Layer256UnitLongerTraining,
        "Mlp3Layer512UnitLongerTraining".lower():Mlp3Layer512UnitLongerTraining,
        "Mlp3Layer768UnitLongerTraining".lower():Mlp3Layer768UnitLongerTraining,

        "Mlp3Layer64UnitLongerTrainingDecreaseBatchSize".lower():Mlp3Layer64UnitLongerTrainingDecreaseBatchSize,
        "Mlp3Layer128UnitLongerTrainingDecreaseBatchSize".lower():Mlp3Layer128UnitLongerTrainingDecreaseBatchSize,
        "Mlp3Layer256UnitLongerTrainingDecreaseBatchSize".lower():Mlp3Layer256UnitLongerTrainingDecreaseBatchSize,
        "Mlp3Layer512UnitLongerTrainingDecreaseBatchSize".lower():Mlp3Layer512UnitLongerTrainingDecreaseBatchSize,
        "Mlp3Layer768UnitLongerTrainingDecreaseBatchSize".lower():Mlp3Layer768UnitLongerTrainingDecreaseBatchSize,


        "CnnA3".lower():CnnA3,
        "CnnA6".lower():CnnA6,
        "CnnAshort".lower():CnnAshort,

        "CnnC3".lower():CnnC3,
        "CnnC6".lower():CnnC6,
        "CnnC3_3".lower():CnnC3_3,

        "Cnn1DC3".lower():Cnn1DC3,
        "Cnn1DC6".lower():Cnn1DC6,
        "Cnn1DC8".lower():Cnn1DC8,

        "Cnn1DC3_Tanh".lower():Cnn1DC3_Tanh,
        "Cnn1DC8_Tanh".lower():Cnn1DC8_Tanh,

        "Cnn1DC3_C".lower():Cnn1DC3_C,
        "Cnn1DC3_CTanh".lower():Cnn1DC3_CTanh,

        "Cnn1DC8_C".lower():Cnn1DC8_C,
        "Cnn1DC8_CTanh".lower():Cnn1DC8_CTanh,

        "Cnn1DC3_D".lower():Cnn1DC3_D,
        "Cnn1DC3_DTanh".lower():Cnn1DC3_DTanh,

        "Cnn1DC8_D".lower():Cnn1DC8_D,
        "Cnn1DC8_DTanh".lower():Cnn1DC8_DTanh,


    }

    return architectures[name.lower()]
