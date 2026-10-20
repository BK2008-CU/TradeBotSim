"""Thin wrapper around Alpaca's trading API. Paper trading by default."""

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


class Broker:
    def __init__(self, key: str, secret: str, paper: bool = True):
        self.paper = paper
        self.client = TradingClient(key, secret, paper=paper)

    def account(self) -> dict:
        a = self.client.get_account()
        return {
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
        }

    def positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in self.client.get_all_positions()
        ]

    def buy_dollars(self, symbol: str, dollars: float):
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(dollars, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(req)

    def close(self, symbol: str):
        return self.client.close_position(symbol)