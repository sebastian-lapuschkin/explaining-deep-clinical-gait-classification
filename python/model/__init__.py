#SVM models
from .svm import SvmLinearL2C1e0
from .svm import SvmLinearL2C1em1
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

#CNN-A models (reading all input channels at once, moving over time axis)
from .cnn import CnnA3
from .cnn import CnnA6
from .cnn import CnnAshort

#CNN-C models (2d-convolutional models)
from .cnn import CnnC3
from .cnn import CnnC6
from .cnn import CnnC3_3

#all below classes need to be registered below and vice versa in order to use the creator-pattern
def get_architecture(name):
    architectures = {
        'SvmLinearL2C1e0'.lower():SvmLinearL2C1e0,
        'SvmLinearL2C1em1'.lower():SvmLinearL2C1em1,
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

        "CnnA3".lower():CnnA3,
        "CnnA6".lower():CnnA6,
        "CnnAshort".lower():CnnAshort,

        "CnnC3".lower():CnnC3,
        "CnnC6".lower():CnnC6,
        "CnnC3_3".lower():CnnC3_3
    }

    return architectures[name.lower()]