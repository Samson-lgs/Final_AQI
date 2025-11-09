import requests
import psycopg2
from datetime import datetime

# --- Database connection details ---
DB_HOST = "ep-patient-breeze-adt1xy0o-pooler.c-2.us-east-1.aws.neon.tech"
DB_NAME = "neondb"
DB_USER = "neondb_owner"
DB_PASSWORD = "npg_1yEJuba0dzxi"

# --- OpenWeather API key ---
API_KEY = "1b4c8511f7820d5f184d63ab205690a0"

# --- List of cities ---
CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna", "Vadodara", "Ludhiana", "Agra", "Nashik", "Faridabad",
    "Meerut", "Rajkot", "Kalyan", "Vasai", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Navi Mumbai",
    "Allahabad", "Ranchi", "Howrah", "Coimbatore", "Jabalpur", "Gwalior", "Vijayawada", "Madurai", "Raipur", "Kota",
    "Guwahati", "Chandigarh", "Hubli", "Tiruchirappalli", "Mysuru", "Bareilly", "Aligarh", "Tiruppur", "Moradabad", "Jalandhar"
]

# --- Connect to PostgreSQL ---
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    sslmode="require"
)
cur = conn.cursor()

# --- Fetch and store data ---
for city in CITIES:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            pressure = data["main"]["pressure"]
            weather_desc = data["weather"][0]["description"]

            cur.execute("""
                INSERT INTO openweather_data (city, temperature, humidity, pressure, weather_description, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (city, temp, humidity, pressure, weather_desc, datetime.now()))

            print(f"✅ Inserted data for {city}")
        else:
            print(f"⚠️ Skipped {city}: {data.get('message', 'No data')}")
    except Exception as e:
        print(f"❌ Error for {city}: {e}")

conn.commit()
cur.close()
conn.close()

print("\n✅ All done — data saved to Neon DB!")
