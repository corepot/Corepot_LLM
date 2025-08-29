import unittest
from app.corepot_model import build_model

class TestCorepotModel(unittest.TestCase):
    def test_model_output(self):
        model = build_model('resnet18', num_classes=10)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()














