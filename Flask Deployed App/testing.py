import unittest
import os
import sys
import tempfile
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

# Add current directory to path so we can import CNN
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import CNN
import pandas as pd


class TestCSVDataLoading(unittest.TestCase):
    """Test that CSV files load correctly and contain expected data."""

    @classmethod
    def setUpClass(cls):
        """Load CSV files once for all tests."""
        cls.disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
        cls.supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

    def test_disease_info_has_39_rows(self):
        """Disease info should have exactly 39 entries (one per class)."""
        self.assertEqual(len(self.disease_info), 39,
                         "disease_info.csv should contain 39 disease entries")

    def test_supplement_info_has_39_rows(self):
        """Supplement info should have exactly 39 entries."""
        self.assertEqual(len(self.supplement_info), 39,
                         "supplement_info.csv should contain 39 supplement entries")

    def test_disease_info_has_required_columns(self):
        """Disease info must have all required columns."""
        required_columns = ['disease_name', 'description', 'Possible Steps', 'image_url']
        for col in required_columns:
            self.assertIn(col, self.disease_info.columns,
                          f"Missing required column: {col}")

    def test_supplement_info_has_required_columns(self):
        """Supplement info must have all required columns."""
        required_columns = ['supplement name', 'supplement image', 'buy link']
        for col in required_columns:
            self.assertIn(col, self.supplement_info.columns,
                          f"Missing required column: {col}")

    def test_no_null_disease_names(self):
        """All disease names should be non-null."""
        self.assertFalse(self.disease_info['disease_name'].isnull().any(),
                         "Disease names contain null values")


class TestClassMapping(unittest.TestCase):
    """Test the CNN class index-to-name mapping."""

    def test_idx_to_classes_has_39_entries(self):
        """There should be exactly 39 class mappings."""
        self.assertEqual(len(CNN.idx_to_classes), 39,
                         "idx_to_classes should have 39 entries")

    def test_all_indices_are_contiguous(self):
        """Class indices should be 0 through 38."""
        expected_indices = set(range(39))
        actual_indices = set(CNN.idx_to_classes.keys())
        self.assertEqual(expected_indices, actual_indices,
                         "Class indices must be 0..38 without gaps")

    def test_class_names_are_unique(self):
        """All class names should be unique."""
        class_names = list(CNN.idx_to_classes.values())
        self.assertEqual(len(class_names), len(set(class_names)),
                         "Class names should be unique")


class TestImagePreprocessing(unittest.TestCase):
    """Test the image preprocessing logic used before prediction."""

    def setUp(self):
        """Create a small test image."""
        self.test_image = Image.new('RGB', (300, 300), color='red')

    def test_image_resize_to_224(self):
        """Image should be resized to 224x224."""
        resized = self.test_image.resize((224, 224))
        self.assertEqual(resized.size, (224, 224),
                         "Image must be resized to 224x224")

    def test_resized_image_is_rgb(self):
        """Resized image should still be RGB mode."""
        resized = self.test_image.resize((224, 224))
        self.assertEqual(resized.mode, 'RGB',
                         "Resized image must be in RGB mode")

    def test_different_input_sizes(self):
        """Images of different sizes should all resize to 224x224."""
        sizes = [(100, 100), (500, 300), (1024, 768), (50, 200)]
        for w, h in sizes:
            img = Image.new('RGB', (w, h), color='green')
            resized = img.resize((224, 224))
            self.assertEqual(resized.size, (224, 224),
                             f"Failed for input size ({w}, {h})")


class TestModelStructure(unittest.TestCase):
    """Test the CNN model structure."""

    def test_model_has_conv_layers(self):
        """Model should have convolutional layers defined."""
        model = CNN.CNN(39)
        self.assertIsNotNone(model.conv_layers,
                             "Model must have conv_layers")
        # Check that conv layers contain Conv2d modules
        has_conv = any(isinstance(layer, nn.Conv2d) for layer in model.conv_layers)
        self.assertTrue(has_conv, "conv_layers should contain Conv2d modules")

    def test_model_output_shape(self):
        """Model should output a vector of size 39."""
        model = CNN.CNN(39)
        # Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 39),
                         f"Expected output shape (1, 39), got {output.shape}")

    def test_model_output_is_logits(self):
        """Raw model output should not be probabilities (no softmax)."""
        model = CNN.CNN(39)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        # Logits can be any value, not bounded by [0, 1]
        output_array = output.detach().numpy()
        self.assertFalse(np.all((output_array >= 0) & (output_array <= 1)),
                         "Raw output should be logits, not probabilities")


class TestTorchTensorConversion(unittest.TestCase):
    """Test torchvision tensor conversion."""

    def test_pil_to_tensor_shape(self):
        """PIL image converted to tensor should have shape (3, 224, 224)."""
        import torchvision.transforms.functional as TF
        img = Image.new('RGB', (224, 224), color='blue')
        tensor = TF.to_tensor(img)
        self.assertEqual(tensor.shape, (3, 224, 224),
                         f"Expected shape (3, 224, 224), got {tensor.shape}")

    def test_tensor_values_in_range(self):
        """Tensor values from to_tensor should be in [0, 1]."""
        import torchvision.transforms.functional as TF
        img = Image.new('RGB', (224, 224), color='red')
        tensor = TF.to_tensor(img)
        self.assertGreaterEqual(tensor.min().item(), 0.0,
                                "Tensor values should be >= 0")
        self.assertLessEqual(tensor.max().item(), 1.0,
                             "Tensor values should be <= 1")


if __name__ == '__main__':
    unittest.main()
