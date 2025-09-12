# Outlier Detection for ALS Implicit Feedback Recommender Systems

## Overview

This guide explains how to use the comprehensive outlier detection system for your ALS implicit feedback recommender. The system is designed to identify and remove various types of outliers that can negatively impact recommendation quality, including:

- **Bot behavior**: Automated users with high-frequency, regular interaction patterns
- **Random clickers**: Users who click on products randomly without genuine interest
- **Suspicious patterns**: Users with unusual timing, low diversity, or other anomalous behaviors
- **Statistical outliers**: Interactions that fall outside normal statistical distributions

## Quick Start

### Basic Usage

```python
from utils.als import process_homepage_and_similar_products_recommendations

# Enable outlier detection with default ecommerce configuration
process_homepage_and_similar_products_recommendations(
    raw_interactions, 
    raw_products, 
    enable_outlier_detection=True
)
```

### Advanced Usage

```python
from utils.als import process_homepage_and_similar_products_recommendations

# Use specific outlier detection configuration
process_homepage_and_similar_products_recommendations(
    raw_interactions, 
    raw_products, 
    enable_outlier_detection=True,
    outlier_config_name="bot_detection"  # Focus on bot detection
)
```

## Available Configurations

### 1. Default Configuration
- **Use case**: General purpose, balanced approach
- **Characteristics**: Moderate outlier detection
- **Best for**: Most e-commerce applications

### 2. Conservative Configuration
- **Use case**: When you want to preserve as much data as possible
- **Characteristics**: Only removes obvious outliers
- **Best for**: Small datasets or when data quality is generally good

### 3. Aggressive Configuration
- **Use case**: When data quality is poor or you have many bots
- **Characteristics**: Removes many potential outliers
- **Best for**: Large datasets with significant noise

### 4. E-commerce Configuration (Default)
- **Use case**: Optimized for typical e-commerce user behavior
- **Characteristics**: Balanced thresholds for online shopping patterns
- **Best for**: Online stores, marketplaces

### 5. Bot Detection Configuration
- **Use case**: Specifically targeting bot traffic
- **Characteristics**: Focuses on automated behavior patterns
- **Best for**: When you suspect significant bot activity

### 6. Mobile App Configuration
- **Use case**: Mobile app interactions
- **Characteristics**: Accounts for faster mobile interaction patterns
- **Best for**: Mobile applications

## Outlier Detection Methods

### 1. User-Level Outlier Detection

Detects users with suspicious behavior patterns:

- **High frequency users**: Users with >100 interactions per day (configurable)
- **Bot-like users**: Users with very regular, fast interaction patterns
- **Low diversity users**: Users who click on the same products repeatedly
- **Suspicious timing users**: Users with unusual interaction timing patterns

### 2. Session-Level Outlier Detection

Identifies suspicious user sessions:

- **Short sessions**: Sessions <30 seconds (potential bot behavior)
- **High product sessions**: Sessions with >50 different products (random clicking)

### 3. Interaction-Level Outlier Detection

Uses statistical methods to detect outlier interactions:

- **Z-score outliers**: Interactions with Z-score >3.0 (configurable)
- **IQR outliers**: Interactions outside 1.5×IQR range (configurable)

## Configuration Parameters

### User-Level Parameters

```python
max_user_interactions_per_day = 100      # Max interactions per user per day
min_user_session_duration_seconds = 30   # Min session duration
max_user_products_per_session = 50       # Max products per session
user_interaction_velocity_threshold = 10 # Max interactions per minute
```

### Statistical Parameters

```python
z_score_threshold = 3.0        # Z-score threshold for outliers
iqr_multiplier = 1.5          # IQR multiplier for outlier detection
```

### Pattern Detection Parameters

```python
min_diversity_ratio = 0.1                    # Min ratio of unique products
max_repeat_interaction_ratio = 0.8           # Max ratio of repeated interactions
suspicious_timing_threshold = 5              # Max seconds between interactions
bot_like_pattern_threshold = 0.9             # Similarity threshold for bot patterns
min_human_like_delay = 1                     # Min delay for human-like behavior
```

## Custom Configuration

### Create Custom Configuration

```python
from utils.outlier_config import create_custom_config

# Create custom configuration
custom_config = create_custom_config(
    max_user_interactions_per_day=150,
    z_score_threshold=2.5,
    min_diversity_ratio=0.15
)

# Use in outlier detection
from utils.outlier_detection import comprehensive_outlier_detection
filtered_interactions, stats = comprehensive_outlier_detection(
    interactions, 
    custom_config
)
```

### Extend Configuration Class

```python
from utils.outlier_detection import OutlierDetectionConfig

class MyCustomConfig(OutlierDetectionConfig):
    def __init__(self):
        super().__init__()
        self.max_user_interactions_per_day = 200
        self.z_score_threshold = 2.0
        # Add other custom parameters
```

## Integration Examples

### 1. Homepage Recommendations

```python
from utils.als import process_homepage_and_similar_products_recommendations

# With outlier detection
process_homepage_and_similar_products_recommendations(
    raw_interactions, 
    raw_products, 
    enable_outlier_detection=True,
    outlier_config_name="ecommerce"
)
```

### 2. Cross-Sell Recommendations

```python
from utils.als import process_cross_sell_recommendation

# With bot detection focus
process_cross_sell_recommendation(
    raw_interactions, 
    raw_products, 
    enable_outlier_detection=True,
    outlier_config_name="bot_detection"
)
```

### 3. Individual Outlier Detection Methods

```python
from utils.outlier_detection import detect_user_outliers, detect_session_outliers

# User-level detection only
filtered_interactions, user_stats = detect_user_outliers(interactions)

# Session-level detection only
filtered_interactions, session_stats = detect_session_outliers(interactions)
```

## Monitoring and Analysis

### Outlier Detection Statistics

The system provides detailed statistics about detected outliers:

```python
from utils.outlier_detection import get_outlier_detection_summary

# Get human-readable summary
summary = get_outlier_detection_summary(outlier_stats)
print(summary)
```

### Example Output

```
Outlier Detection Summary:
  Original interactions: 50,000
  Final interactions: 42,500
  Total removed: 7,500
  Removal rate: 15.00%

User-level outliers:
  High frequency users: 25
  Bot-like users: 12
  Low diversity users: 8
  Suspicious timing users: 15
  Total outlier users: 45

Session-level outliers:
  Short sessions: 120
  High product sessions: 85
  Total outlier sessions: 180

Interaction-level outliers:
  Z-score outliers: 1,200
  IQR outliers: 800
  Total outlier interactions: 1,500
```

## Best Practices

### 1. Start Conservative
Begin with the conservative configuration and gradually increase strictness based on your data quality.

### 2. Monitor Removal Rates
Aim for 5-15% removal rate. Higher rates may indicate overly aggressive settings.

### 3. Validate Results
Check that removed users/interactions are actually outliers by examining their patterns.

### 4. A/B Testing
Compare recommendation quality with and without outlier detection to measure impact.

### 5. Regular Review
Periodically review and adjust outlier detection parameters as your user base evolves.

## Troubleshooting

### Too Many Outliers Detected
- Use conservative configuration
- Increase thresholds (e.g., `max_user_interactions_per_day`)
- Check if your data has genuine quality issues

### Too Few Outliers Detected
- Use aggressive configuration
- Decrease thresholds
- Consider bot detection configuration

### Performance Issues
- Disable outlier detection for initial testing
- Use individual detection methods selectively
- Consider processing data in batches

## Example Scripts

See `examples/outlier_detection_example.py` for comprehensive examples of:
- Different configuration presets
- Individual detection methods
- Custom configuration creation
- Integration with ALS training

## API Reference

### Main Functions

- `comprehensive_outlier_detection()`: Complete outlier detection pipeline
- `detect_user_outliers()`: User-level outlier detection
- `detect_session_outliers()`: Session-level outlier detection
- `detect_interaction_outliers()`: Interaction-level outlier detection

### Configuration Functions

- `get_outlier_config()`: Get predefined configuration
- `create_custom_config()`: Create custom configuration

### Utility Functions

- `get_outlier_detection_summary()`: Generate summary statistics