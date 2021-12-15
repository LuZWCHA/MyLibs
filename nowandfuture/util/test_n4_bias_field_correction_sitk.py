import glob
import os
from unittest import TestCase

from nowandfuture.util import plot
from nowandfuture.util.preproccess import n4_bias_field_correction_sitk as n4b
import nowandfuture.util.preproccess as prp
import numpy as np

class TestN4_bias_field_correction_sitk(TestCase):
    def test_n4_bias_field_correction_sitk(self):

        self.assertIsInstance(res, np.ndarray)
