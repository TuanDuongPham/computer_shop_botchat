import json
import time
from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL, PRODUCT_CATEGORIES, BATCH_SIZE, MAX_BATCH_ATTEMPTS


class ProductGenerator:
    def __init__(self, postgres_db, chroma_db):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.postgres_db = postgres_db
        self.chroma_db = chroma_db
        self.all_products = set()

    def _get_prompt_for_category(self, category, batch_number):
        category_specs = {
            "CPU": """
                - socket: Socket type (e.g., AM5, LGA1700)
                - cores: Number of cores
                - threads: Number of threads
                - base_clock: Base clock speed in GHz
                - boost_clock: Boost clock speed in GHz
                - cache: Cache size and levels
                - tdp: Thermal Design Power in watts
                - architecture: CPU architecture and generation
                - integrated_graphics: Yes/No and model if applicable
                            """,
            "Motherboard": """
                - chipset: Chipset model
                - socket: CPU socket type
                - form_factor: ATX, mATX, ITX, etc.
                - memory_slots: Number of memory slots
                - max_memory: Maximum supported memory in GB
                - memory_type: DDR4/DDR5 and supported speeds
                - pcie_slots: Full list of PCIe slots and generations
                - sata_ports: Number of SATA ports
                - m2_slots: Number and specs of M.2 slots
                - usb_ports: Types and counts of USB ports
                - wifi: Built-in WiFi specifications if available
                - bluetooth: Bluetooth version if available
                            """,
            "RAM": """
                - type: Memory type (DDR4, DDR5)
                - capacity: Total capacity in GB
                - speed: Memory speed in MHz
                - cas_latency: CAS latency
                - modules: Number of modules and size per module
                - rgb: Whether it has RGB lighting
                - heat_spreader: Heat spreader details
                            """,
            "PSU": """
                - wattage: Power output in watts
                - efficiency: 80+ rating (Bronze, Gold, Platinum, etc.)
                - modularity: Full, semi, or non-modular
                - fan_size: Fan size in mm
                - protection_features: List of protection features
                - connectors: Detailed list of all available connectors with counts
                            """,
            "GPU": """
                - chip: GPU chip model
                - memory: VRAM amount in GB
                - memory_type: Memory type (GDDR6, GDDR6X, etc.)
                - base_clock: Base clock in MHz
                - boost_clock: Boost clock in MHz
                - cuda_cores: For NVIDIA, or stream processors for AMD
                - rt_cores: Ray tracing cores if applicable
                - tdp: Power consumption in watts
                - power_connectors: Required power connectors
                - display_outputs: List of display outputs with counts
                - length: Card length in mm
                            """,
            "Storage": """
                - type: SSD/HDD/NVMe
                - capacity: Storage capacity in GB/TB
                - interface: SATA/NVMe/PCIe
                - form_factor: 2.5", M.2, 3.5", etc.
                - read_speed: Sequential read speed in MB/s
                - write_speed: Sequential write speed in MB/s
                - cache: Cache size if applicable
                - tbw: Terabytes Written endurance rating for SSDs
                            """,
            "Case": """
                - form_factor: Supported motherboard sizes
                - dimensions: Physical dimensions (HxWxD) in mm
                - drive_bays: Number and types of drive bays
                - expansion_slots: Number of expansion slots
                - front_io: Front I/O ports
                - cooling_support: Fan and radiator mounting options
                - gpu_clearance: Maximum GPU length supported in mm
                - cpu_cooler_clearance: Maximum CPU cooler height in mm
                - psu_clearance: PSU size limitations
                            """,
            "Cooling": """
                - type: Air cooler/AIO liquid cooler
                - fan_size: Fan dimensions in mm
                - fan_count: Number of fans
                - radiator_size: For liquid coolers, radiator dimensions
                - rpm_range: Fan RPM range
                - noise_level: Noise level in dBA
                - socket_compatibility: List of compatible CPU sockets
                - tdp_rating: Maximum TDP rating in watts
            """
        }

        prompt = f"""Generate {BATCH_SIZE} REALISTIC and DIVERSE {category} products in JSON format. Each product MUST BE UNIQUE with NO DUPLICATES.

            For {category}, include these DETAILED specifications:
            {category_specs.get(category, "")}

            Format each product with these fields:
            - name: Full product name
            - brand: Manufacturer name
            - model: Model number/name (MUST be unique)
            - price: Price in USD (numeric, realistic market price)
            - specs: DETAILED specifications object based on the category-specific details above
            - stock: Available units (integer between 5-30)

            IMPORTANT: These products should be COMPLETELY DIFFERENT from any generated in previous batches. Create unique models with different specifications than before.

            Return ONLY a valid JSON array with these {BATCH_SIZE} products, each with highly detailed specs. ENSURE the specs are compatible with modern PC components being sold in 2024.

            This is batch #{batch_number} for {category}. Make sure all products have different specifications than previously generated batches.
            """
        return prompt

    def _extract_json_from_response(self, text):
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        return text

    def _flatten_specs(self, specs):
        specs_flat = []
        for key, value in specs.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    specs_flat.append(f"{key}_{subkey}: {subvalue}")
            else:
                specs_flat.append(f"{key}: {value}")

        return ". ".join(specs_flat)

    def generate_products(self, products_per_category=100):
        self.postgres_db.insert_categories(PRODUCT_CATEGORIES)
        category_ids = self.postgres_db.get_category_ids(PRODUCT_CATEGORIES)

        for category in PRODUCT_CATEGORIES:
            if category in ["PSU", "Case", "Cooling"]:
                products_per_category = 50

            print(
                f"Generating {products_per_category} products for {category}...")
            category_id = category_ids[category]
            products_created = 0

            batch_number = 1
            while products_created < products_per_category:
                print(f"  Processing batch {batch_number} for {category}...")

                try:
                    prompt = self._get_prompt_for_category(
                        category, batch_number)

                    response = self.client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8 + (batch_number * 0.02),
                        max_tokens=3000,
                    )

                    products_json = response.choices[0].message.content
                    products_json = self._extract_json_from_response(
                        products_json)

                    try:
                        products = json.loads(products_json)

                        for product in products:
                            product_identifier = f"{product['brand']}_{product['model']}"

                            # Skip if duplicate
                            if product_identifier in self.all_products:
                                print(
                                    f"  Skipping duplicate product: {product['name']}")
                                continue

                            self.all_products.add(product_identifier)

                            # Insert product into PostgreSQL
                            product_id = self.postgres_db.insert_product(
                                category_id, product)

                            # Process specs for ChromaDB
                            specs_text = self._flatten_specs(product['specs'])

                            # Add to ChromaDB
                            self.chroma_db.add_product(
                                product_id, product, category, specs_text)

                            products_created += 1
                            if products_created >= products_per_category:
                                break

                        print(
                            f"  Added {min(len(products), products_per_category - (products_created - len(products)))} new products for {category}, batch {batch_number}")

                    except json.JSONDecodeError as e:
                        print(
                            f"  Error decoding JSON for {category}, batch {batch_number}: {e}")
                        print(f"  Raw JSON: {products_json[:100]}...")

                except Exception as e:
                    print(
                        f"  Error generating products for {category}, batch {batch_number}: {e}")
                    time.sleep(2)

                batch_number += 1

                if batch_number > MAX_BATCH_ATTEMPTS:
                    print(
                        f"  Reached maximum batch attempts for {category}. Generated {products_created} products.")
                    break

            print(
                f"Completed generating {products_created} products for {category}")

        return self.all_products
