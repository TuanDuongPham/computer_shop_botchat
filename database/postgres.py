import json
import psycopg2
from config import POSTGRES_CONFIG


class PostgresDB:
    def __init__(self):
        self.conn = None
        self.cur = None

    def connect(self):
        self.conn = psycopg2.connect(
            dbname=POSTGRES_CONFIG["dbname"],
            user=POSTGRES_CONFIG["user"],
            password=POSTGRES_CONFIG["password"],
            host=POSTGRES_CONFIG["host"],
            port=POSTGRES_CONFIG["port"]
        )
        self.cur = self.conn.cursor()
        print("Connected to PostgreSQL database")
        return self

    def create_tables(self):
        self.cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'categories'
        )
        """)
        tables_exist = self.cur.fetchone()[0]

        if not tables_exist:
            print("Creating database tables for first run...")
            self.cur.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50) UNIQUE NOT NULL
            )
            """)

            self.cur.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                category_id INTEGER REFERENCES categories(id),
                name VARCHAR(100) NOT NULL,
                brand VARCHAR(50) NOT NULL,
                model VARCHAR(50) NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                specs JSONB NOT NULL,
                stock INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            self.conn.commit()
            print("Database tables created successfully")
        else:
            print("Database tables already exist, skipping creation")

    def insert_categories(self, categories):
        for category in categories:
            self.cur.execute(
                "INSERT INTO categories (name) VALUES (%s) ON CONFLICT DO NOTHING",
                (category,)
            )
        self.conn.commit()

    def get_category_ids(self, categories):
        category_ids = {}
        for category in categories:
            self.cur.execute(
                "SELECT id FROM categories WHERE name = %s", (category,))
            category_ids[category] = self.cur.fetchone()[0]
        return category_ids

    def insert_product(self, category_id, product):
        self.cur.execute("""
        INSERT INTO products (category_id, name, brand, model, price, specs, stock)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """, (
            category_id,
            product["name"],
            product["brand"],
            product["model"],
            product["price"],
            json.dumps(product["specs"]),
            product["stock"]
        ))
        product_id = self.cur.fetchone()[0]
        self.conn.commit()
        return product_id

    def get_product_count_by_category(self):
        self.cur.execute("""
        SELECT categories.name, COUNT(*) 
        FROM products 
        JOIN categories ON products.category_id = categories.id 
        GROUP BY categories.name
        """)
        return self.cur.fetchall()

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("PostgreSQL connection closed")
