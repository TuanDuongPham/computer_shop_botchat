def generate_enhanced_product_document(product, category, detailed_specs):
    """
    Generate a rich, detailed document for a product with expanded synonyms
    and structured information to improve embedding quality.

    Args:
        product: The product dictionary
        category: The product category
        detailed_specs: Detailed specifications as text

    Returns:
        A well-structured document string for embedding
    """
    # Category-specific synonyms and terminology to include
    category_synonyms = {
        "CPU": ["processor", "central processing unit", "microprocessor", "chip"],
        "Motherboard": ["mainboard", "system board", "logic board", "mobo"],
        "RAM": ["memory", "random access memory", "system memory", "DIMM"],
        "PSU": ["power supply", "power unit", "power source"],
        "GPU": ["graphics card", "video card", "graphics adapter", "display adapter"],
        "Storage": ["drive", "disk", "memory", "data storage"],
        "Case": ["chassis", "enclosure", "tower", "cabinet", "housing"],
        "Cooling": ["thermal solution", "heatsink", "fan", "temperature management"]
    }

    # Add alternative terms for this category
    synonyms = category_synonyms.get(category, [])
    category_terms = f"{category} " + " ".join(synonyms)

    # Format price with different representations
    price = product["price"]
    price_text = f"${price} {price} USD {price} dollars price:{price}"

    # Build a rich document with structured sections for better semantic search
    document = f"""
        PRODUCT: {product['name']} {product['brand']} {product['model']}
        CATEGORY: {category_terms}
        BRAND: {product['brand']}
        MODEL: {product['model']}
        PRICE: {price_text}
        SPECIFICATIONS:
        {detailed_specs}
    """

    # Add additional sections based on category
    if category == "CPU":
        cores = product["specs"].get("cores", "")
        threads = product["specs"].get("threads", "")
        document += f"\nPERFORMANCE: {cores} cores {threads} threads processor performance computing power"

    elif category == "GPU":
        memory = product["specs"].get("memory", "")
        document += f"\nGRAPHICS PERFORMANCE: {memory}GB VRAM graphics performance gaming rendering visualization"

    elif category == "Storage":
        capacity = product["specs"].get("capacity", "")
        document += f"\nSTORAGE CAPACITY: {capacity} data storage space disk space"

    # Add compatibility information
    if "socket" in product["specs"]:
        socket = product["specs"]["socket"]
        document += f"\nCOMPATIBILITY: {socket} compatible works with supports"

    # Add stock information
    document += f"\nAVAILABILITY: {product['stock']} in stock available units"

    return document.strip()
