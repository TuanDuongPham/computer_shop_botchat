USD_TO_VND_RATE = 25000


def convert_usd_to_vnd(price_usd):
    if not price_usd:
        return 0

    price_usd = float(price_usd)
    return price_usd * USD_TO_VND_RATE


def format_price_vnd(price_vnd, include_currency=True):
    price_vnd = float(price_vnd)
    formatted_price = "{:,.0f}".format(price_vnd).replace(",", ".")

    if include_currency:
        return f"{formatted_price}đ"
    return formatted_price


def format_price_usd_to_vnd(price_usd, include_currency=True):
    price_vnd = convert_usd_to_vnd(price_usd)
    return format_price_vnd(price_vnd, include_currency)


def parse_vnd_price(price_string):
    price_string = price_string.replace("đ", "").replace(
        "₫", "").replace("VND", "").strip()
    price_string = price_string.replace(".", "")

    return float(price_string)


def parse_usd_from_vnd(price_vnd):
    price_vnd = float(price_vnd)
    return price_vnd / USD_TO_VND_RATE


def detect_and_convert_price(price_input):
    if isinstance(price_input, str):
        if any(currency in price_input.lower() for currency in ["$", "usd", "dollar"]):
            try:
                price_value = float(price_input.replace("$", "").replace(
                    "USD", "").replace("dollar", "").strip())
                price_vnd = convert_usd_to_vnd(price_value)
                return price_vnd, format_price_vnd(price_vnd)
            except (ValueError, TypeError):
                return 0, "0đ"
        else:
            price_vnd = parse_vnd_price(price_input)
            return price_vnd, format_price_vnd(price_vnd)
    else:
        price_vnd = convert_usd_to_vnd(price_input)
        return price_vnd, format_price_vnd(price_vnd)
