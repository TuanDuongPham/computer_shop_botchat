class SharedStateService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedStateService, cls).__new__(cls)
            cls._instance.init_state()
        return cls._instance

    def init_state(self):
        self.recently_advised_products = []
        self.recently_advised_pc = False
        self.session_data = {}

    def set_recently_advised_products(self, products):
        self.recently_advised_products = products

        pc_components = ["CPU", "Motherboard", "RAM", "GPU", "Storage", "PSU"]
        found_components = [product.get("category")
                            for product in products if "category" in product]

        if len(set(found_components).intersection(pc_components)) >= 4:
            self.recently_advised_pc = True
            print("Đã lưu trữ cấu hình PC vừa tư vấn")
        else:
            self.recently_advised_pc = False

        print(
            f"SharedStateService: Đã lưu trữ {len(products)} sản phẩm tư vấn gần nhất")

    def get_recently_advised_products(self):
        return self.recently_advised_products

    def is_recently_advised_pc(self):
        return self.recently_advised_pc

    def set_session_data(self, key, value):
        self.session_data[key] = value

    def get_session_data(self, key, default=None):
        return self.session_data.get(key, default)
