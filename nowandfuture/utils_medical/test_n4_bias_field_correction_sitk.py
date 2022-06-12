import glob
import os
from unittest import TestCase

from nowandfuture.utils_medical import plot
from nowandfuture.utils_medical.preproccess import n4_bias_field_correction_sitk as n4b
import nowandfuture.utils_medical.preproccess as prp
import numpy as np

class TestN4_bias_field_correction_sitk(TestCase):
    def test_n4_bias_field_correction_sitk(self):

        self.assertIsInstance(res, np.ndarray)
