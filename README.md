# ALS Implicit Recommender System

This project documents my journey of building a production-ready recommender system, starting from a naive prototype and evolving into a scalable, cloud-optimized ML pipeline.

What began as a simple "statistical model running on fake data" quickly turned into a deep engineering challenge once real production data arrived. This repository contains the code, pipelines, and tooling that came out of that process.

## Overview
The first version of the model was built in a few days:
- Local CPU-only training
- Fake synthetic data
- Deployed on Azure as a proof-of-concept

It worked… until real data entered. Suddenly there were **millions of interactions, messy schemas, mixed types, and inconsistent column names**. From this point on, the real project began.

## Key Challenges and Solutions
### 1. Data Cleaning at Scale
Real production data was far from clean. I built a full preprocessing pipeline that:
- Normalizes and aligns inconsistent schemas
- Fixes mismatched data types
- Removes corrupted or incomplete rows
- Validates field formats

Data cleaning turned out to be more demanding than the model itself.

### 2. Vectorization and Performance Bottlenecks
The first implementation was **too slow** for large datasets.
To fix this, I:
- Vectorized heavy operations
- Profiled performance hotspots
- Removed Python loops and rewrote them using NumPy logic

This reduced training time, **locally**, from **infinite** to **minutes**.

### 3. Understanding the Algorithm (ALS in Implicit)
Once the data was finally clean, the results were still bad.
That's when I realized I could no longer treat the model as a "black box", I had to understand what was actually happening under the hood.

I used the excellent [benfred/implicit](https://github.com/benfred/implicit) library, specifically the Alternating Least Squares (ALS) implementation. 
The first version "worked" but I didn't understand why it worked or why it wasn't working well on my real data.

To fix that, I had to learn:
- How ALS factorization actually works
- Why implicit feedback requires confidence weights
- The role of matrix multiplication
- What each hyperparameter actually does and their trade-offs
- How to run systematic parameter searches

If I need to single out one paper that helped me the most with the general understanding it's: [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf).

This shifted the model from a "black box" to something I could reason about.

### 4. Production Workflow Optimization
Fetching the **entire dataset for every train was inefficient**.
I redesigned the process to:
- Keep the previous year of training data inside Azure blob storage
- Fetch only the last 24 hours of new interactions

**This drastically reduced:**
- Database load
- Production server load
- Training time

### 5. Cloud Migration and Cost Efficiency
I originally picked Azure because of my .NET background, but I quickly realized it wasn't the right fit.
Even though CPU training wasn't that slow on my local machine, Azure CPU runs were extremely slow, mostly because I was using the cheaper App Service tier, while everything faster was unreasonably expensive.
**Instead of scaling up on Azure, I migrated to Google Cloud with GPU support**, where:
- Training went from **multi-hour** → **2 minutes**
- GPU instances were much more affordable
- The environment felt easier to work with

Azure now only handles the daily scheduling, while all heavy lifting happens on GCP.

### 6. In-Memory vs Distributed Caching
Originally everything was stored in in-memory hash maps, not scalable.
I moved to Redis, and the production recommendation flow became:

`API → userId → Redis → Top 20 recommendations → API`

### 7. Real-Time vs Offline Training
A real-time updating model was unrealistic after analyzing:
- Compute cost
- Complexity
- Required freshness

Switched to offline training with scheduled model updates, simpler and more reliable.

### 8. Debugging "Good Metrics, Bad Results"
Even with low loss, results looked off.
Digging deeper uncovered:

- Thousands of bot interactions (Google/Facebook crawlers)
- Internal admin activity polluting the dataset

Cleaning this noise improved results more than any hyperparameter tweak.
