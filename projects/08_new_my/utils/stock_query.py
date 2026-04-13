"""股票实时行情查询（新浪免费接口，支持 A 股、港股、美股）"""

import re
import requests
from urllib.parse import quote

SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
}

COMMON_STOCK_MAPPING = {
    "贵州茅台": "sh600519",
    "平安银行": "sz000001",
    "招商银行": "sh600036",
    "阿里巴巴": "hk09988",
    "阿里巴巴美股": "gb_baba",
    "腾讯": "hk00700",
    "腾讯控股": "hk00700",
    "苹果": "gb_aapl",
    "特斯拉": "gb_tsla",
    "英伟达": "gb_nvda",
    "微软": "gb_msft",
}


def _safe_float(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _format_number(value, unit=""):
    if value is None:
        return "--"
    abs_value = abs(value)
    if abs_value >= 100000000:
        return f"{value / 100000000:.2f}亿{unit}"
    if abs_value >= 10000:
        return f"{value / 10000:.2f}万{unit}"
    return f"{value:.2f}{unit}" if isinstance(value, float) else f"{value}{unit}"


def _search_stock_code(keyword: str):
    search_url = f"https://suggest3.sinajs.cn/suggest/type=11,12,13,14,15&key={quote(keyword)}&name=suggestdata"
    try:
        response = requests.get(search_url, headers=SINA_HEADERS, timeout=5)
        response.encoding = "gbk"
        match = re.search(r'="([^"]*)"', response.text)
        if not match or not match.group(1):
            return None
        for item in match.group(1).split(";"):
            parts = item.split(",")
            if len(parts) < 4:
                continue
            code = parts[3].strip().lower()
            if re.fullmatch(r"(sh|sz)\d{6}|hk\d{5}|gb_[a-z.]+", code):
                return code
    except Exception:
        pass
    return None


def _normalize_stock_code(query: str):
    text = query.strip()
    lowered = text.lower()
    if not text:
        return None
    if text in COMMON_STOCK_MAPPING:
        return COMMON_STOCK_MAPPING[text]
    if re.fullmatch(r"(sh|sz)\d{6}|hk\d{5}|gb_[a-z.]+", lowered):
        return lowered
    if re.fullmatch(r"\d{6}", text):
        return f"sh{text}" if text.startswith(("5", "6", "9")) else f"sz{text}"
    if re.fullmatch(r"\d{5}", text):
        return f"hk{text}"
    if re.fullmatch(r"[A-Za-z.]{1,10}", text):
        return f"gb_{lowered}"
    return _search_stock_code(text)


def _fetch_stock_fields(code: str):
    try:
        response = requests.get(f"https://hq.sinajs.cn/list={code}", headers=SINA_HEADERS, timeout=5)
        response.encoding = "gbk"
        response.raise_for_status()
        match = re.search(r'="([^"]*)"', response.text.strip())
        if not match or not match.group(1):
            return None
        fields = [item.strip() for item in match.group(1).split(",")]
        return fields if any(fields) else None
    except Exception:
        return None


def _parse_a_stock(code: str, fields):
    if len(fields) < 32:
        return None
    pre_close = _safe_float(fields[2])
    price = _safe_float(fields[3])
    change = price - pre_close if price and pre_close else None
    change_pct = change / pre_close * 100 if change and pre_close else None
    return {
        "market": "A股", "name": fields[0], "code": code, "price": price,
        "open": _safe_float(fields[1]), "pre_close": pre_close,
        "high": _safe_float(fields[4]), "low": _safe_float(fields[5]),
        "change": change, "change_pct": change_pct,
        "volume": _safe_float(fields[8]), "turnover": _safe_float(fields[9]),
        "currency": "CNY", "updated_at": f"{fields[30]} {fields[31]}".strip(),
    }


def _parse_hk_stock(code: str, fields):
    if len(fields) < 18:
        return None
    updated_at = fields[17]
    if len(fields) > 18 and fields[18]:
        updated_at = f"{fields[17]} {fields[18]}"
    return {
        "market": "港股", "name": fields[1] or fields[0], "code": code,
        "price": _safe_float(fields[6]), "open": _safe_float(fields[2]),
        "pre_close": _safe_float(fields[3]), "high": _safe_float(fields[4]),
        "low": _safe_float(fields[5]), "change": _safe_float(fields[7]),
        "change_pct": _safe_float(fields[8]),
        "volume": _safe_float(fields[12]) if len(fields) > 12 else None,
        "turnover": _safe_float(fields[11]) if len(fields) > 11 else None,
        "currency": "HKD", "updated_at": updated_at.strip(),
    }


def _parse_us_stock(code: str, fields):
    if len(fields) < 11:
        return None
    pre_close = _safe_float(fields[26]) if len(fields) > 26 else None
    price = _safe_float(fields[1])
    change = _safe_float(fields[4]) if len(fields) > 4 else None
    if change is None and price and pre_close:
        change = price - pre_close
    change_pct = _safe_float(fields[2]) if len(fields) > 2 else None
    if change_pct is None and change and pre_close:
        change_pct = change / pre_close * 100
    return {
        "market": "美股", "name": fields[0], "code": code, "price": price,
        "open": _safe_float(fields[5]) if len(fields) > 5 else None,
        "pre_close": pre_close,
        "high": _safe_float(fields[6]) if len(fields) > 6 else None,
        "low": _safe_float(fields[7]) if len(fields) > 7 else None,
        "change": change, "change_pct": change_pct,
        "volume": _safe_float(fields[10]) if len(fields) > 10 else None,
        "turnover": None, "market_cap": _safe_float(fields[12]) if len(fields) > 12 else None,
        "currency": "USD", "updated_at": fields[3] if len(fields) > 3 else "",
    }


def _parse_stock_quote(code: str, fields):
    if code.startswith(("sh", "sz")):
        return _parse_a_stock(code, fields)
    if code.startswith("hk"):
        return _parse_hk_stock(code, fields)
    if code.startswith("gb_"):
        return _parse_us_stock(code, fields)
    return None


def query_stock(keyword: str) -> str:
    """查询股票实时行情，返回格式化字符串"""
    keyword = keyword.strip()
    if not keyword:
        return "请输入股票名称或代码。"

    try:
        code = _normalize_stock_code(keyword)
        if not code:
            return f"未识别股票：{keyword}。\n可尝试更完整的名称或直接输入代码，如：600519、hk00700、AAPL。"

        fields = _fetch_stock_fields(code)
        if not fields:
            return f"未查询到 {keyword} 的行情数据。"

        stock = _parse_stock_quote(code, fields)
        if not stock:
            return f"{keyword} 的数据格式暂不支持解析。"

        # 格式化输出
        price_text = f'{stock["price"]:.2f} {stock["currency"]}' if stock["price"] else "--"
        change_text = "--"
        pct_text = "--"
        if stock["change"] is not None:
            sign = "+" if stock["change"] > 0 else ""
            change_text = f"{sign}{stock['change']:.2f}"
        if stock["change_pct"] is not None:
            sign = "+" if stock["change_pct"] > 0 else ""
            pct_text = f"{sign}{stock['change_pct']:.2f}%"

        lines = [
            f"📈 {stock['name']}（{stock['market']} / {stock['code']}）",
            f"当前价：{price_text}",
            f"开盘：{stock['open'] or '--'} | 最高：{stock['high'] or '--'} | 最低：{stock['low'] or '--'}",
            f"涨跌：{change_text} ({pct_text})",
        ]
        if stock["volume"] is not None:
            lines.append(f"成交量：{_format_number(stock['volume'], '股')}")
        if stock["turnover"] is not None:
            lines.append(f"成交额：{_format_number(stock['turnover'])} {stock['currency']}")
        if stock.get("market_cap") is not None:
            lines.append(f"总市值：{_format_number(stock['market_cap'])} {stock['currency']}")
        if stock["updated_at"]:
            lines.append(f"更新时间：{stock['updated_at']}")

        return "\n".join(lines)

    except requests.exceptions.RequestException:
        return f"查询失败：网络请求错误"
    except Exception as e:
        return f"查询失败：{e}"
