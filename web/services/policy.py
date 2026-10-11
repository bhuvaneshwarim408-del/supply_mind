"""Replays a reorder policy against a demand series (used for the projected stock chart)."""


def simulate(stock0, demand, reorder_point, order_qty, lead_days):
    stock = float(stock0)
    pending = []  # (arrival_day, qty)
    series, orders = [], 0
    short = total = 0.0
    for i, d in enumerate(demand):
        stock += sum(q for a, q in pending if a == i)
        pending = [(a, q) for a, q in pending if a != i]
        d = max(0.0, float(d))
        sold = min(stock, d)
        stock -= sold
        short += d - sold
        total += d
        if stock <= reorder_point and not pending and order_qty > 0:
            pending.append((i + lead_days, order_qty))
            orders += 1
        series.append(round(stock, 1))
    service = round(100 * (1 - short / total), 1) if total > 0 else 100.0
    return {"stock": series, "orders": orders, "service_pct": service}
