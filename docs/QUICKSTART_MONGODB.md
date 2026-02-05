```bash
pip install -r requirements.txt
```

```bash
docker-compose up -d
```

```bash
python scripts/load_data_to_mongodb.py
```

Otvori u browser-u: **http://localhost:8081**
- Username: `admin`
- Password: `admin123`

### MongoDB direktno
```bash
# Koristeći mongo shell
mongosh "mongodb://admin:admin123@localhost:27017"

# Prebaci se na bazu
use pa_recommender

# Proveri kolekcije
show collections

# Primer upita
db.interactions.countDocuments()
db.product_recommendations.countDocuments()
```

## Pokretanje analitičkih upita

```bash
jupyter notebook examples/mongodb_analytics.ipynb
```

---

## Zaustavljanje servisa

```bash
docker-compose down
```

Za brisanje svih podataka:
```bash
docker-compose down -v
```

---