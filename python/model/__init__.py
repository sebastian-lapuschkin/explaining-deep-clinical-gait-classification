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
