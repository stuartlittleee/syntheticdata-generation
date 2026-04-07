# syntheticdata-generation

## AeroTrust training data generator

Generate synthetic aerospace maintenance narratives for ModernBERT classifier training.

### Requirements

- Python 3.9+
- pandas

Install:

```bash
pip install pandas
```

### Usage

```bash
python /home/runner/work/syntheticdata-generation/syntheticdata-generation/generate_aerotrust_training_data.py --rows 1200 --seed 42 --output /home/runner/work/syntheticdata-generation/syntheticdata-generation/aerotrust_training_data.csv
```

The script outputs a CSV with exactly these columns:

- `raw_narrative`
- `system_category` (`Avionics`, `Hydraulics`, `Structural`, `Propulsion`, `Cabin/ECS`)
- `risk_level` (`Low`, `Medium`, `Critical`)
