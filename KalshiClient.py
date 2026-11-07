"""
Kalshi API client — class wrapper around auth, market data, orders, and
portfolio endpoints. Import KalshiClient into other scripts (e.g. an
autoquoter) rather than duplicating request/auth logic.
"""
from __future__ import annotations

import os

import base64
import time
import uuid
from dataclasses import dataclass

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    def __init__(self, key_id: str, private_key_path: str,
                 base_url: str = "https://api.elections.kalshi.com"):
        self.key_id = key_id
        self.base_url = base_url
        with open(private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)

    # -----------------------------------------------------------------
    # AUTH / LOW-LEVEL REQUEST
    # -----------------------------------------------------------------
    def _sign(self, method: str, path: str, ts_ms: str) -> str:
        msg = f"{ts_ms}{method}{path}".encode("utf-8")
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def request(self, method: str, endpoint: str, params=None, json_body=None, auth: bool = True):
        full_path = "/trade-api/v2" + endpoint
        url = self.base_url + full_path
        headers = {"Content-Type": "application/json"}
        if auth:
            ts = str(int(time.time() * 1000))
            headers.update({
                "KALSHI-ACCESS-KEY": self.key_id,
                "KALSHI-ACCESS-SIGNATURE": self._sign(method, full_path, ts),
                "KALSHI-ACCESS-TIMESTAMP": ts,
            })
        resp = requests.request(method, url, headers=headers, params=params, json=json_body)
        if not resp.ok:
            print(f"KALSHI ERROR {resp.status_code} on {method} {full_path}: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    # -----------------------------------------------------------------
    # MARKET DATA (public, no auth needed, but harmless to sign anyway)
    # -----------------------------------------------------------------
    def find_markets(self, series_ticker: str, status: str = "open") -> list[dict]:
        resp = self.request("GET", "/markets", params={"series_ticker": series_ticker, "status": status})
        return resp.get("markets", [])

    def get_market(self, ticker: str) -> dict:
        """
        Fetches a single market's full metadata directly from Kalshi —
        including close_time and floor_strike/cap_strike. IMPORTANT:
        close_time is when the market STOPS ACCEPTING ORDERS, not
        necessarily when the underlying condition is actually evaluated —
        those can be (and for this contract type, are) different moments.
        For the actual strike/resolution observation time, use get_event()
        and read its "strike_date" field instead.
        """
        resp = self.request("GET", f"/markets/{ticker}", auth=False)
        return resp.get("market", resp)

    def get_event(self, event_ticker: str) -> dict:
        """
        Fetches event-level metadata, including "strike_date" — the actual
        moment the underlying condition is evaluated (e.g. "DXY above strike
        at 11:59 PM ET"). This is the authoritative resolution-observation
        time; do NOT use the market's close_time for this purpose, since
        close_time can be (and often is) later — a buffer/dispute window
        after the real observation moment, not the observation moment itself.
        """
        resp = self.request("GET", f"/events/{event_ticker}", auth=False)
        return resp.get("event", resp)

    def get_orderbook(self, ticker: str, depth: int = 5) -> dict:
        resp = self.request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth}, auth=False)
        return resp.get("orderbook_fp", {})

    @staticmethod
    def best_market_prices(book: dict) -> tuple[int, int] | None:
        """Returns (best_yes_bid_cents, best_yes_ask_cents) or None if thin."""
        yes_levels = book.get("yes_dollars", [])
        no_levels = book.get("no_dollars", [])
        if not yes_levels or not no_levels:
            return None
        best_yes_bid = round(max(float(p) for p, _ in yes_levels) * 100)
        best_no_bid = round(max(float(p) for p, _ in no_levels) * 100)
        best_yes_ask = 100 - best_no_bid
        return best_yes_bid, best_yes_ask

    def get_market_mid_prob(self, ticker: str) -> float | None:
        """Mid price between best YES bid and best YES ask, as a 0-1 probability."""
        book = self.get_orderbook(ticker)
        prices = self.best_market_prices(book)
        if prices is None:
            return None
        best_yes_bid, best_yes_ask = prices
        return ((best_yes_bid + best_yes_ask) / 2) / 100.0

    # -----------------------------------------------------------------
    # ORDERS (V2)
    # -----------------------------------------------------------------
    def place_limit_order(self, ticker: str, side: str, price_cents: int, count: int,
                           post_only: bool = True) -> dict:
        """
        side: "yes" or "no". Translated internally to Kalshi's V2 bid/ask
        model — buying NO at price P is economically identical to selling
        YES (ask) at price (1 - P).

        post_only=True (default): the order is cancelled instead of filled
        if it would cross the book, guaranteeing you only ever add liquidity.
        """
        if side == "yes":
            v2_side, price_dollars = "bid", price_cents / 100
        elif side == "no":
            v2_side, price_dollars = "ask", (100 - price_cents) / 100
        else:
            raise ValueError(f"Unknown side: {side}")

        body = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "side": v2_side,
            "count": f"{count:.2f}",
            "price": f"{price_dollars:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": post_only,
        }
        resp = self.request("POST", "/portfolio/events/orders", json_body=body)
        if resp.get("remaining_count") == "0.00" and resp.get("fill_count", "0.00") == "0.00":
            print(f"  WARNING: post_only order for {ticker} {side} @ {price_cents}c was "
                  f"rejected (would have crossed the book) — not resting.")
        return resp

    def cancel_order(self, order_id: str) -> dict:
        return self.request("DELETE", f"/portfolio/events/orders/{order_id}")

    def get_order(self, order_id: str) -> dict:
        """Fetches a single order by ID — used by amend_order to retrieve the
        original client_order_id, which Kalshi's amend endpoint requires."""
        resp = self.request("GET", f"/portfolio/orders/{order_id}")
        return resp.get("order", resp)

    def amend_order(self, order_id: str, ticker: str, side: str, price_cents: int,
                     count: int) -> dict:
        """
        Atomically modifies an EXISTING order's price and/or size in a
        single call — no cancel, no new order, same order_id throughout.
        This is the correct tool for "reprice a resting quote": there is
        never a moment with zero coverage (order never disappears) and
        never a moment with two orders resting at once (it's still the
        same one order, just repriced).

        side: "yes" or "no", same external interface as place_limit_order —
        translated internally to Kalshi's V2 bid/ask model.

        Amending to a smaller size preserves queue position; amending to a
        larger size or a different price does not (per Kalshi's docs).
        """
        if side == "yes":
            v2_side, price_dollars = "bid", price_cents / 100
        elif side == "no":
            v2_side, price_dollars = "ask", (100 - price_cents) / 100
        else:
            raise ValueError(f"Unknown side: {side}")

        # Kalshi's amend endpoint requires the order's ORIGINAL client_order_id,
        # which we don't retain after placing it — fetch it fresh rather than
        # guessing, since a wrong value here could get the amend rejected.
        original = self.get_order(order_id)
        original_client_order_id = original.get("client_order_id")

        body = {
            "ticker": ticker,
            "side": v2_side,
            "price": f"{price_dollars:.4f}",
            "count": f"{count:.2f}",
            "client_order_id": original_client_order_id,
            "updated_client_order_id": str(uuid.uuid4()),
        }
        return self.request("POST", f"/portfolio/events/orders/{order_id}/amend", json_body=body)

    def cancel_orders_batch(self, order_ids: list[str], chunk_size: int = 20) -> None:
        """
        Cancels multiple orders in a single request each, chunked to a safe
        batch size, instead of one HTTP round-trip per order. Kalshi's docs
        note this doesn't save API tokens (same total cost), but it collapses
        N sequential network round-trips into far fewer, which is what
        actually matters for wall-clock speed (each round-trip is ~50-200ms).
        Falls back to individual cancel_order calls for any chunk that fails
        as a batch, so a single bad order_id can't block the rest.
        """
        if not order_ids:
            return
        for i in range(0, len(order_ids), chunk_size):
            chunk = order_ids[i:i + chunk_size]
            body = {"orders": [{"order_id": oid, "subaccount": 0, "exchange_index": 0} for oid in chunk]}
            try:
                self.request("DELETE", "/portfolio/events/orders/batched", json_body=body)
            except requests.HTTPError:
                # Fall back to individual cancels for this chunk only
                for oid in chunk:
                    try:
                        self.cancel_order(oid)
                    except requests.HTTPError as e:
                        if getattr(e.response, "status_code", None) == 404:
                            continue  # already gone — not an error worth logging
                        print(f"  Failed to cancel {oid}: {e}")

    def get_open_orders(self, ticker: str | None = None) -> list[dict]:
        params = {"ticker": ticker} if ticker else None
        resp = self.request("GET", "/portfolio/orders", params=params)
        return resp.get("orders", [])

    @staticmethod
    def economic_side(o: dict) -> str | None:
        """
        Determines the ECONOMIC direction of an order (yes vs no), which is
        NOT the same as the raw "side" field. Kalshi's book is single and
        YES-denominated: a "no" order is placed as an ask (sell YES) at the
        complementary price, and gets reported back under the YES side of
        the book. Reading the raw "side" field alone will misidentify a NO
        order as a YES position (and price it at 1-minus-what-you-think).

        book_side == "bid"  -> economically YES (buying YES)
        book_side == "ask"  -> economically NO (selling YES == buying NO)

        Falls back to the raw "side" field if book_side isn't present
        (shouldn't normally happen, but avoids silently returning None).
        """
        book_side = o.get("book_side")
        if book_side == "bid":
            return "yes"
        elif book_side == "ask":
            return "no"
        return o.get("side")

    @staticmethod
    def economic_price_field(o: dict) -> str:
        """Which price field holds the economically-correct price for this
        order, given its true (book_side-derived) economic direction."""
        return "yes_price_dollars" if KalshiClient.economic_side(o) == "yes" else "no_price_dollars"

    def get_my_resting_price(self, ticker: str, side: str) -> tuple[int | None, float]:
        """Returns (price_cents, total_remaining_size) for the price with the
        largest remaining size on a given ECONOMIC side. (None, 0.0) if
        nothing resting."""
        by_price: dict[int, float] = {}
        for o in self.get_open_orders(ticker):
            if self.economic_side(o) != side or not self._is_actually_resting(o):
                continue
            pd_ = o.get(self.economic_price_field(o))
            if not pd_:
                continue
            pc = round(float(pd_) * 100)
            by_price[pc] = by_price.get(pc, 0.0) + float(o.get("remaining_count_fp", 0))
        if not by_price:
            return None, 0.0
        best_price = max(by_price, key=lambda k: by_price[k])
        return best_price, by_price[best_price]

    @staticmethod
    def _is_actually_resting(o: dict) -> bool:
        """
        Kalshi's legacy GET /portfolio/orders listing can return orders that
        have already been fully filled/expired/cancelled server-side (status
        stays "resting" in the listing but remaining_count_fp is 0, or the
        order simply no longer exists — DELETE returns 404). Filter those out
        before attempting to cancel, instead of retrying dead order IDs on
        every single loop forever.
        """
        return o.get("status") == "resting" and float(o.get("remaining_count_fp", 0)) > 0

    def pull_side(self, ticker: str, side: str) -> float:
        """Cancels all genuinely-resting orders on one ECONOMIC side of this
        ticker (skips already-dead/ghost entries), via a single batched
        cancel call instead of one request per order. Returns total size pulled."""
        orders = [o for o in self.get_open_orders(ticker)
                  if self.economic_side(o) == side and self._is_actually_resting(o)]
        if not orders:
            return 0.0
        total = sum(float(o.get("remaining_count_fp", 0)) for o in orders)
        self.cancel_orders_batch([o["order_id"] for o in orders])
        return total

    def cancel_all_orders(self, ticker: str) -> None:
        orders = [o for o in self.get_open_orders(ticker) if self._is_actually_resting(o)]
        self.cancel_orders_batch([o["order_id"] for o in orders])

    # -----------------------------------------------------------------
    # PORTFOLIO
    # -----------------------------------------------------------------
    def get_balance(self) -> dict:
        return self.request("GET", "/portfolio/balance")

    def get_positions(self) -> list[dict]:
        resp = self.request("GET", "/portfolio/positions")
        return resp.get("market_positions", [])


# ===========================================================================
# STANDALONE MAIN — same interactive pull -> Enter -> requote demo as before,
# now built on the KalshiClient class. Run this file directly to test the
# class against your live account without importing it into anything else.
# ===========================================================================
if __name__ == "__main__":
    KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
    PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
    TICKER = "KXDXYDUD-26JUL13-T100.9650"              # <-- update daily
    SIDE = "yes"
    PRICE_CENTS = 60

    client = KalshiClient(key_id=KEY_ID, private_key_path=PRIVATE_KEY_PATH)

    print("=== PORTFOLIO ===")
    print("Balance:", client.get_balance())
    print("Positions:", client.get_positions())

    print(f"\n=== YOUR OPEN ORDERS ON {TICKER} ===")
    for o in client.get_open_orders(TICKER):
        side = o.get("side")
        pd_ = o.get("yes_price_dollars") if side == "yes" else o.get("no_price_dollars")
        pc = round(float(pd_) * 100) if pd_ else None
        print(f"  order_id={o.get('order_id')} side={side} price={pc}c "
              f"remaining={o.get('remaining_count_fp')} status={o.get('status')}")

    print(f"\nPulling {SIDE} @ {PRICE_CENTS}c...")
    size = client.pull_side(TICKER, SIDE)
    # pull_side cancels ALL resting orders on that side, regardless of price —
    # if you only want to pull one specific price level, filter first via
    # get_my_resting_price(ticker, side) and cancel_order() individually instead.
    print(f"Pulled {size} total contracts on {SIDE} side.")

    input("\nPress Enter to requote...")

    if size > 0:
        print(f"Requoting {int(size)} @ {PRICE_CENTS}c on {SIDE} side...")
        result = client.place_limit_order(TICKER, side=SIDE, price_cents=PRICE_CENTS, count=int(size))
        print("Quote result:", result)
    else:
        print("Nothing to requote — no size was pulled.")
