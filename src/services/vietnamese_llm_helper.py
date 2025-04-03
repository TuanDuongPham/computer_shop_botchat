from openai import OpenAI
from src.config import OPENAI_API_KEY

CATEGORY_TRANSLATIONS = {
    "CPU": ["Nhân", "Vi xử lý", "Bộ xử lý", "Core", "Processor", "Chip", "CPU Intel", "CPU AMD", "Xử lý", "Xử lý trung tâm"],
    "Motherboard": ["Bo mạch chủ", "Mainboard", "Main"],
    "RAM": ["RAM", "Bộ nhớ", "Memory"],
    "SSD": ["Ổ cứng thể rắn", "Ổ SSD", "Solid State Drive"],
    "PSU": ["Nguồn", "Power Supply"],
    "GPU": ["Card đồ họa", "Card", "VGA", "Graphics Card"],
    "HDD": ["Ổ cứng", "Hard Drive", "HDD"],
    "Storage": ["Lưu trữ", "Ổ cứng"],
    "Cooling": ["Tản nhiệt", "Quạt", "Cooler"],
    "Case": ["Vỏ máy tính", "Thùng máy", "Case"],
}

COMMON_BRANDS = {
    "CPU": ["Intel", "AMD", "Ryzen", "Core i3", "Core i5", "Core i7", "Core i9", "Xeon", "Pentium", "Celeron"],
    "GPU": ["NVIDIA", "AMD", "RTX", "GTX", "Radeon"],
    "Motherboard": ["ASUS", "Gigabyte", "MSI", "ASRock"],
    "RAM": ["Corsair", "Kingston", "G.Skill", "Crucial", "ADATA", "TeamGroup"],
    "Storage": ["Samsung", "Western Digital", "Seagate", "Crucial", "Kingston"],
}

SPEC_MAPPINGS = {
    "CPU": {
        "socket": ["socket", "đế cắm"],
        "cores": ["nhân", "cores", "lõi"],
        "threads": ["luồng", "threads"],
        "base_clock": ["xung cơ bản", "xung nhịp", "tần số"],
        "boost_clock": ["xung tăng cường", "xung turbo", "turbo boost"],
        "cache": ["bộ nhớ đệm", "cache"],
        "tdp": ["công suất", "tdp", "điện năng tiêu thụ"],
        "architecture": ["kiến trúc", "architecture", "công nghệ"],
        "integrated_graphics": ["đồ họa tích hợp", "igpu", "gpu tích hợp"],
    },
    "Motherboard": {
        "chipset": ["chipset", "chip điều khiển"],
        "socket": ["socket", "đế cắm"],
        "form_factor": ["kích thước", "form factor", "chuẩn bo mạch"],
        "memory_slots": ["khe ram", "số khe ram", "slots ram"],
        "max_memory": ["ram tối đa", "dung lượng ram tối đa"],
        "memory_type": ["loại ram", "kiểu ram", "chuẩn ram"],
        "pcie_slots": ["khe pcie", "slot pcie", "cổng pcie"],
        "sata_ports": ["cổng sata", "số cổng sata"],
        "m2_slots": ["khe m.2", "slot m.2", "số khe m.2"],
        "usb_ports": ["cổng usb", "số cổng usb"],
        "wifi": ["wifi", "không dây", "wireless"],
        "bluetooth": ["bluetooth", "kết nối bluetooth"],
    },
    "RAM": {
        "type": ["loại", "chuẩn", "kiểu"],
        "capacity": ["dung lượng", "capacity", "kích thước"],
        "speed": ["tốc độ", "bus", "tần số"],
        "cas_latency": ["độ trễ", "cas latency", "timing"],
        "modules": ["thanh", "số thanh", "số lượng"],
        "rgb": ["rgb", "đèn led", "led"],
        "heat_spreader": ["tản nhiệt", "heatsink", "tản nhiệt kim loại"],
    },
    "PSU": {
        "wattage": ["công suất", "watt", "w"],
        "efficiency": ["hiệu suất", "80 plus", "chuẩn hiệu suất"],
        "modularity": ["module", "dạng dây", "kiểu dây"],
        "fan_size": ["kích thước quạt", "quạt", "đường kính quạt"],
        "protection_features": ["bảo vệ", "chống chập", "chống quá tải"],
        "connectors": ["đầu nối", "connector", "cổng kết nối"],
    },
    "GPU": {
        "chip": ["chip đồ họa", "gpu", "vi mạch"],
        "memory": ["vram", "bộ nhớ", "dung lượng"],
        "memory_type": ["loại bộ nhớ", "chuẩn bộ nhớ", "kiểu vram"],
        "base_clock": ["xung nhịp cơ bản", "xung cơ bản", "tần số cơ bản"],
        "boost_clock": ["xung tăng cường", "xung boost", "tần số tối đa"],
        "cuda_cores": ["nhân cuda", "cuda cores", "đơn vị xử lý"],
        "rt_cores": ["nhân rt", "rt cores", "ray tracing"],
        "tdp": ["công suất", "mức tiêu thụ điện", "điện năng"],
        "power_connectors": ["đầu cấp nguồn", "kết nối nguồn", "cổng nguồn"],
        "display_outputs": ["cổng xuất hình", "kết nối màn hình", "output"],
        "length": ["chiều dài", "kích thước", "độ dài"],
    },
    "Storage": {
        "type": ["loại", "kiểu", "chuẩn"],
        "capacity": ["dung lượng", "kích thước", "không gian lưu trữ"],
        "interface": ["chuẩn kết nối", "giao tiếp", "interface"],
        "form_factor": ["kích thước", "form factor", "quy cách"],
        "read_speed": ["tốc độ đọc", "đọc", "băng thông đọc"],
        "write_speed": ["tốc độ ghi", "ghi", "băng thông ghi"],
        "cache": ["bộ nhớ đệm", "cache", "buffer"],
        "tbw": ["độ bền", "tuổi thọ", "endurance"],
    },
    "Case": {
        "form_factor": ["kích thước", "chuẩn bo mạch", "hỗ trợ bo mạch"],
        "dimensions": ["kích thước vỏ", "chiều cao chiều rộng chiều dài", "dimensions"],
        "drive_bays": ["khay ổ cứng", "vị trí ổ cứng", "số khay"],
        "expansion_slots": ["khe mở rộng", "slot", "số khe"],
        "front_io": ["cổng trước", "đầu nối mặt trước", "front panel"],
        "cooling_support": ["hỗ trợ tản nhiệt", "gắn quạt", "số quạt"],
        "gpu_clearance": ["chiều dài card đồ họa", "độ dài gpu", "clearance gpu"],
        "cpu_cooler_clearance": ["chiều cao tản nhiệt cpu", "độ cao tản cpu", "clearance cpu"],
        "psu_clearance": ["không gian nguồn", "vị trí nguồn", "psu shroud"],
    },
    "Cooling": {
        "type": ["loại", "kiểu", "chuẩn"],
        "fan_size": ["kích thước quạt", "đường kính quạt", "size"],
        "fan_count": ["số lượng quạt", "số quạt", "quạt"],
        "radiator_size": ["kích thước radiator", "tản nhiệt nước", "rad size"],
        "rpm_range": ["tốc độ quay", "vòng quay", "rpm"],
        "noise_level": ["độ ồn", "tiếng ồn", "decibel"],
        "socket_compatibility": ["tương thích socket", "hỗ trợ socket", "socket"],
        "tdp_rating": ["khả năng tản nhiệt", "công suất tản nhiệt", "cooling capacity"],
    },
}


class VietnameseLLMHelper:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def enhance_vietnamese_query(self, query):
        # Chuẩn bị từ điển danh mục để đưa vào prompt
        category_mappings = ""
        for english_term, vietnamese_terms in CATEGORY_TRANSLATIONS.items():
            vietnamese_list = ", ".join(vietnamese_terms)
            category_mappings += f"- {english_term}: {vietnamese_list}\n"

        # Chuẩn bị từ điển thương hiệu để đưa vào prompt
        brand_mappings = ""
        for category, brands in COMMON_BRANDS.items():
            brand_list = ", ".join(brands)
            brand_mappings += f"- {category}: {brand_list}\n"

        # Chuẩn bị từ điển thông số kỹ thuật để đưa vào prompt
        spec_mappings_text = ""
        for category, specs in SPEC_MAPPINGS.items():
            spec_mappings_text += f"\n{category}:\n"
            for spec_name, spec_terms in specs.items():
                spec_terms_str = ", ".join(spec_terms)
                spec_mappings_text += f"  - {spec_name}: {spec_terms_str}\n"

        prompt = f"""
            Bạn là một chuyên gia song ngữ Việt-Anh về phần cứng máy tính.
            Một người dùng vừa nhập truy vấn bằng tiếng Việt để tìm kiếm linh kiện máy tính.
            Hãy cải thiện truy vấn này bằng cách thêm các thuật ngữ kỹ thuật tiếng Anh tương ứng để giúp tìm kiếm
            các sản phẩm phù hợp trong cơ sở dữ liệu chứa thông tin bằng tiếng Anh.

            Truy vấn tiếng Việt: "{query}"

            DANH MỤC SẢN PHẨM:
            {category_mappings}
            
            THƯƠNG HIỆU PHỔ BIẾN:
            {brand_mappings}

            THÔNG SỐ KỸ THUẬT CHO TỪNG DANH MỤC:
            {spec_mappings_text}

            Quy tắc:
            1. Phân tích truy vấn để xác định người dùng đang tìm kiếm danh mục sản phẩm nào
            2. Thêm các thuật ngữ tiếng Anh tương ứng từ danh sách DANH MỤC SẢN PHẨM
            3. Xác định các thông số kỹ thuật được đề cập trong truy vấn và thêm các thuật ngữ tiếng Anh tương ứng từ danh sách THÔNG SỐ KỸ THUẬT
            4. Nếu truy vấn chứa giá tiền, hãy chuyển đổi nó thành USD (1 triệu VND = 40 USD)
            5. Nếu truy vấn chứa từ "chip", bạn phải nhận diện đó là đang nói đến "CPU" hoặc "processor"
            6. Chỉ trả về truy vấn đã được nâng cao mà KHÔNG có bất kỳ giải thích nào, chỉ trả về văn bản truy vấn

            Ví dụ 1:
            Cho "tản nhiệt nước cho CPU Intel socket LGA1700" bạn có thể trả về:
            "Cooling water cooling liquid cooler AIO cooler Intel LGA1700 socket compatibility"

            Ví dụ 2:
            Cho "cần card đồ họa 8GB VRAM chơi game 1440p" bạn có thể trả về:
            "GPU Graphics Card VGA memory 8GB GDDR6 gaming performance 1440p"

            Ví dụ 3:
            Cho "bo mạch chủ hỗ trợ RAM DDR5 6000MHz và nhiều khe M.2" bạn có thể trả về:
            "Motherboard mainboard memory type DDR5 speed 6000MHz m2 slots storage expansion"
            
            Ví dụ 4:
            Cho "chip intel core i5 14600K" bạn có thể trả về:
            "CPU processor Intel Core i5 14600K chip high performance"
            
            Ví dụ 5:
            Cho "cần card đồ họa chơi game tốt dưới 5 triệu" bạn có thể trả về:
            "Graphics Card gaming performance budget affordable below 200 USD"
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )

            enhanced_query = response.choices[0].message.content.strip()
            print(f"Enhanced query: \"{enhanced_query}\"")
            return enhanced_query

        except Exception as e:
            print(f"LLM enhancement failed: {e}")
            return query
