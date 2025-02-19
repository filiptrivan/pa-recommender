import json
import random
import faker
import os
import pandas as pd


try:
    from pyspark.sql import SparkSession  # noqa: F401
except ImportError:
    pass  # skip this import if we are in pure python environment

MMLSPARK_PACKAGE = "com.microsoft.azure:synapseml_2.12:0.9.5"
MMLSPARK_REPO = "https://mmlspark.azureedge.net/maven"

def generate_fake_data():
    fake = faker.Faker()
    
    num_users = 1000
    num_products = 100
    num_purchases = 5000
    product_categories = ['Power Tools', 'Hand Tools', 'Gardening Tools', 'Measuring Tools', 'Safety Equipment']
    
    users = []
    for user_id in range(1, num_users + 1):
        users.append({
            "UserID": user_id,
            "Age": random.randint(20, 60),
            "Gender": random.choice(["Male", "Female"]),
            "ShippingAddress": fake.address()
        })
    
    products = []
    for product_id in range(1, num_products + 1):
        products.append({
            "ProductID": product_id,
            "ProductName": fake.word().capitalize() + " " + random.choice(["Drill", "Saw", "Hammer", "Wrench", "Screwdriver"]),
            "Category": random.choice(product_categories),
            "Price": round(random.uniform(10.0, 500.0), 2)
        })
    
    user_purchases = []
    for _ in range(num_purchases):
        user_id = random.randint(1, num_users)
        product_id = random.randint(1, num_products)
        purchase_date = fake.date_this_year()
        review = random.choice([fake.sentence() for _ in range(3)])
        user_purchases.append({
            "UserID": user_id,
            "ProductID": product_id,
            "PurchaseDate": str(purchase_date),
            "Review": review
        })
    
    user_reviews = []
    for purchase in user_purchases:
        if random.random() < 0.8:  # 80% of purchases will have reviews
            rating = random.randint(1, 5)
            review_text = fake.sentence()
            user_reviews.append({
                "UserID": purchase["UserID"],
                "ProductID": purchase["ProductID"],
                "Rating": rating,
                "ReviewText": review_text
            })
    
    dataset = {
        "users": users,
        "products": products,
        "userPurchases": user_purchases,
        "userReviews": user_reviews
    }
    
    # Save to a JSON file
    with open('tools_recommendation_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Dataset saved.")
    
    
def start_or_get_spark(
    app_name="Sample",
    url="local[*]",
    memory="10g",
    config=None,
    packages=None,
    jars=None,
    repositories=None,
):
    """Start Spark if not started

    Args:
        app_name (str): set name of the application
        url (str): URL for spark master
        memory (str): size of memory for spark driver. This will be ignored if spark.driver.memory is set in config.
        config (dict): dictionary of configuration options
        packages (list): list of packages to install
        jars (list): list of jar files to add
        repositories (list): list of maven repositories

    Returns:
        object: Spark context.
    """

    submit_args = ""
    if packages is not None:
        submit_args = "--packages {} ".format(",".join(packages))
    if jars is not None:
        submit_args += "--jars {} ".format(",".join(jars))
    if repositories is not None:
        submit_args += "--repositories {}".format(",".join(repositories))
    if submit_args:
        os.environ["PYSPARK_SUBMIT_ARGS"] = "{} pyspark-shell".format(submit_args)

    spark_opts = [
        'SparkSession.builder.appName("{}")'.format(app_name),
        'master("{}")'.format(url),
    ]

    if config is not None:
        for key, raw_value in config.items():
            value = (
                '"{}"'.format(raw_value) if isinstance(raw_value, str) else raw_value
            )
            spark_opts.append('config("{key}", {value})'.format(key=key, value=value))

    if config is None or "spark.driver.memory" not in config:
        spark_opts.append('config("spark.driver.memory", "{}")'.format(memory))

    # Set larger stack size
    spark_opts.append('config("spark.executor.extraJavaOptions", "-Xss4m")')
    spark_opts.append('config("spark.driver.extraJavaOptions", "-Xss4m")')

    spark_opts.append("getOrCreate()")
    return eval(".".join(spark_opts))

def save_csv_arrays_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load JSON as dictionary
        
    for key, value in data.items():
        if isinstance(value, list): 
            df = pd.DataFrame(value) 
            csv_file = f"{key}.csv" 
            df.to_csv(csv_file, index=False) 
            print(f"Saved {csv_file}")

def load_spark_df(spark, csv_file, size=None, schema=None):
    df = spark.read.csv(csv_file, header=True, schema=schema)
    if size:
        df = df.limit(size)
    return df