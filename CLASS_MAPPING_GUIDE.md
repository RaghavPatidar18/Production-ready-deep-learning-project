# 🎯 Understanding Class Mapping in Keras CNN Classifier

## How Keras Knows Which Class is Which

When using `flow_from_directory()` in Keras, the framework automatically handles class mapping based on **alphabetical order** of folder names.

---

## 🍌 Your Banana Dataset Class Mapping

### Folder Structure:
```
artifacts/data_ingestion/Banana/
├── train/
│   ├── freshripe/
│   ├── freshunripe/
│   ├── overripe/
│   ├── ripe/
│   ├── rotten/
│   └── unripe/
├── valid/
│   ├── freshripe/
│   ├── freshunripe/
│   ├── overripe/
│   ├── ripe/
│   ├── rotten/
│   └── unripe/
└── test/
    ├── freshripe/
    ├── freshunripe/
    ├── overripe/
    ├── ripe/
    ├── rotten/
    └── unripe/
```

### Automatic Class Index Mapping (Alphabetical Order):
```python
{
    0: "freshripe",
    1: "freshunripe",
    2: "overripe",
    3: "ripe",
    4: "rotten",
    5: "unripe"
}
```

---

## 🔄 How It Works Across the Pipeline

### 1. **Training Phase** (`model_trainer.py`)

```python
self.train_generator = train_datagenerator.flow_from_directory(
    directory=self.config.training_data / "train",
    shuffle=True,
    **dataflow_kwargs
)

# Keras automatically creates class_indices attribute:
# train_generator.class_indices = {
#     'freshripe': 0,
#     'freshunripe': 1,
#     'overripe': 2,
#     'ripe': 3,
#     'rotten': 4,
#     'unripe': 5
# }
```

**What happens:**
- Keras scans the `train/` directory
- Finds 6 subdirectories (class folders)
- Assigns indices in **alphabetical order**
- Stores mapping in `train_generator.class_indices`
- **NOW SAVES** this mapping to `artifacts/training/class_indices.json` for later use!

---

### 2. **Evaluation Phase** (`evaluation.py`)

```python
self.valid_generator = valid_datagenerator.flow_from_directory(
    directory=self.config.training_data / "test",
    shuffle=False,
    **dataflow_kwargs
)
```

**What happens:**
- Keras scans the `test/` directory
- Creates the **same mapping** (alphabetical order is consistent)
- Model predictions match the class indices from training
- Evaluation metrics are correctly calculated

**Example Output:**
```
Found 60 images belonging to 6 classes.
Evaluation using class indices: {'freshripe': 0, 'freshunripe': 1, 'overripe': 2, 'ripe': 3, 'rotten': 4, 'unripe': 5}
```

---

### 3. **Prediction Phase** (`predict.py`)

```python
# Model outputs 6 probabilities (one for each class)
predictions = model.predict(test_image)
# Example: [0.05, 0.02, 0.01, 0.85, 0.05, 0.02]
#           ^^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^^^
#            0     1     2     3     4     5

# Get the class with highest probability
predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Returns: 3

# Map index to class name using saved mapping
predicted_class = self.class_mapping[3]  # Returns: "ripe"
```

**What happens:**
1. Image is preprocessed (resize to 224x224, rescale to 0-1)
2. Model predicts probability for each of the 6 classes
3. Index with highest probability is selected
4. Index is mapped to class name using loaded mapping from `class_indices.json`

---

## 🔑 Key Points

### ✅ Why It Works Consistently:

1. **Alphabetical Order is Deterministic**
   - "freshripe" will always be index 0
   - "unripe" will always be index 5
   - Same order during training, validation, testing, and prediction

2. **flow_from_directory() is Consistent**
   - Uses same sorting logic every time
   - Doesn't depend on file system order
   - Python's sorted() function ensures consistency

3. **Model Learns Based on Indices**
   - During training, model learns that index 0 = freshripe patterns
   - During prediction, output[0] still represents freshripe

### ⚠️ Important Warnings:

1. **Never Change Folder Names After Training**
   - Changing "ripe" → "banana_ripe" would break the mapping!
   - Alphabetical order would change

2. **Don't Add/Remove Classes**
   - Model is trained for exactly 6 classes
   - Adding a 7th class requires retraining

3. **Always Use Same Preprocessing**
   - Training uses `rescale=1./255`
   - Prediction MUST also divide by 255.0

---

## 📊 Verification During Runtime

### Training Output:
```
Found 120 images belonging to 6 classes.
Class indices saved: {'freshripe': 0, 'freshunripe': 1, 'overripe': 2, 'ripe': 3, 'rotten': 4, 'unripe': 5}
```

### Evaluation Output:
```
Found 60 images belonging to 6 classes.
Evaluation using class indices: {'freshripe': 0, 'freshunripe': 1, 'overripe': 2, 'ripe': 3, 'rotten': 4, 'unripe': 5}
```

### Prediction Output:
```
Loaded class mapping from file: {0: 'freshripe', 1: 'freshunripe', 2: 'overripe', 3: 'ripe', 4: 'rotten', 5: 'unripe'}
Predicted Class: ripe
Confidence: 94.52%
All probabilities: [0.01 0.02 0.01 0.9452 0.01 0.01]
```

---

## 🛠️ Best Practices Implemented

1. **✅ Save Class Indices During Training**
   - Stored in `artifacts/training/class_indices.json`
   - Used by prediction pipeline for consistency

2. **✅ Load Class Indices During Prediction**
   - No hardcoding of class names
   - Automatically syncs with training

3. **✅ Print Class Mapping During Evaluation**
   - Verification that mappings match
   - Debugging aid

4. **✅ Consistent Preprocessing**
   - Same rescaling (1./255) everywhere
   - Same image size (224x224)

---

## 🧪 How to Test

### Test a single image:
```python
from cnnClassifier.pipeline.predict import PredictionPipeline

# Test with a ripe banana image
predictor = PredictionPipeline("artifacts/data_ingestion/Banana/test/ripe/sample.jpg")
result = predictor.predict()
print(result)
# Expected: [{'image': 'ripe', 'confidence': '92.45%'}]
```

### Test all classes:
```python
import os
from pathlib import Path

test_dir = Path("artifacts/data_ingestion/Banana/test")
for class_name in ['freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe']:
    class_dir = test_dir / class_name
    image_files = list(class_dir.glob("*.jpg"))
    if image_files:
        test_image = str(image_files[0])
        predictor = PredictionPipeline(test_image)
        result = predictor.predict()
        print(f"True class: {class_name}, Predicted: {result[0]['image']}")
```

---

## 📚 Summary

**Question:** How does the model know which class belongs to which?

**Answer:** 
1. Keras `flow_from_directory()` automatically assigns indices based on **alphabetical order** of folder names
2. During **training**, this mapping is created and **saved** to `class_indices.json`
3. During **evaluation**, the same alphabetical order ensures consistent mapping
4. During **prediction**, we load the saved mapping to convert indices back to class names
5. The model outputs probabilities for each index, and we use the mapping to get the class name

**The mapping is deterministic, consistent, and transparent throughout the entire pipeline!** 🎯

