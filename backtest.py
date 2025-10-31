import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import xlsxwriter
import json
import os

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartMoneyScalpingBacktest:
    def __init__(self, symbol: str, initial_balance: float = 1000.0, symbol_config: dict = None):
        self.exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
        self.symbol = symbol
        self.timeframe = '15m'
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # Configuración específica por símbolo
        self.symbol_config = symbol_config or {}
        
        # Parámetros Optimizados para MAYOR FRECUENCIA
        self.structure_lookback = 20
        self.risk_reward_ratio = self.symbol_config.get('risk_reward_ratio', 2)
        self.leverage = 20
        self.risk_per_trade_pct = 0.05 #riesgo por trade
        self.max_candles_in_trade = 24
        # Parámetros Pools de Liquidez (48h ≈ 576 velas de 15m)
        self.pool_lookback_bars = 192  # 48h de velas de 15m
        self.equal_tol = 0.0003  # 0.03%
        self.min_rr = 1.5
        
        # Configuraciones específicas por símbolo
        self.enable_structural_stop = self.symbol_config.get('enable_structural_stop', True)
        self.tp_percentage = self.symbol_config.get('tp_percentage', 2.0)
        self.sl_percentage = self.symbol_config.get('sl_percentage', 1.0)
        
        # Historial
        self.trades = []
        self.equity_curve = [{'time': None, 'balance': self.initial_balance}]
        self.active_trade = None
        # Callback para reportar PnL al cerrar trades (para multisímbolo)
        self.pnl_callback = None  # type: ignore
        # Filtro oscilador (MACD 12,26,9 en 15m)
        self.enable_macd_filter = True
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

    def fetch_historical_data(self, days=10) -> pd.DataFrame:
        """Descarga datos históricos."""
        try:
            logger.info(f"Descargando datos para {self.symbol} ({self.timeframe}) de los últimos {days} días...")
            limit = 1000
            since = self.exchange.milliseconds() - 86400000 * days
            all_ohlcv = []
            
            while since < self.exchange.milliseconds():
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=limit)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(all_ohlcv) % 20000 == 0: logger.info(f"Descargadas {len(all_ohlcv)} velas...")
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Convertir a UTC y luego a la zona horaria deseada (UTC-4)
            # Esta es la nueva línea (convierte a UTC-3)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Etc/GMT+3')
            df.set_index('timestamp', inplace=True)
            logger.info(f"✅ Descarga completa: {len(df)} velas desde {df.index[0]}.")
            return df
        except Exception as e:
            logger.error(f"❌ Error descargando datos: {e}")
            return None

    def find_patterns(self):
        """Identifica la estructura del mercado, barridos de liquidez y FVGs."""
        logger.info("Identificando patrones SMC (alcistas y bajistas)...")
        
        n = self.structure_lookback
        self.df['min'] = self.df.iloc[argrelextrema(self.df.low.values, np.less_equal, order=n)[0]]['low']
        self.df['max'] = self.df.iloc[argrelextrema(self.df.high.values, np.greater_equal, order=n)[0]]['high']

        self.df['is_fvg_bullish'] = False
        self.df['is_fvg_bearish'] = False

        lows = self.df['low'].values
        highs = self.df['high'].values
        is_fvg_bullish = np.zeros(len(self.df), dtype=bool)
        is_fvg_bearish = np.zeros(len(self.df), dtype=bool)

        for i in range(2, len(self.df)):
            if lows[i] > highs[i-2]: is_fvg_bullish[i-1] = True
            if highs[i] < lows[i-2]: is_fvg_bearish[i-1] = True
        
        self.df['is_fvg_bullish'] = is_fvg_bullish
        self.df['is_fvg_bearish'] = is_fvg_bearish

    def compute_atr(self, window: int = 14):
        """Calcula ATR simple para umbrales de proximidad."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=window, min_periods=window).mean()

    def compute_macd(self, fast: int = None, slow: int = None, signal: int = None):
        """Calcula MACD clásico (EMA fast/slow + signal) sobre cierre."""
        if fast is None: fast = self.macd_fast
        if slow is None: slow = self.macd_slow
        if signal is None: signal = self.macd_signal
        close = self.df['close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        self.df['macd'] = macd_line
        self.df['macd_signal'] = macd_signal
        self.df['macd_hist'] = macd_hist

    def _binsize(self, ref_price: float, tol: float) -> float:
        return max(1e-8, ref_price * tol)

    def build_liquidity_pools(self, i: int, lookback: int = None, tol: float = None):
        """Construye pools de liquidez como concentración de niveles en una ventana.
        Suma puntuación por toques repetidos (equal highs/lows), swings, niveles redondos y bordes de FVG.
        Devuelve lista de dicts: {'price': float, 'score': float}.
        """
        if lookback is None:
            lookback = self.pool_lookback_bars
        if tol is None:
            tol = self.equal_tol

        start = max(0, i - lookback)
        window = self.df.iloc[start:i]
        if window.empty:
            return []

        mid_price = float(window['close'].iloc[-1])
        binsize = self._binsize(mid_price, tol)
        step = 5.0 if mid_price < 5000 else 10.0
        pools = {}

        def add(price: float, score: float):
            if price is None or np.isnan(price):
                return
            bucket = round(price / binsize)
            level = bucket * binsize
            if level in pools:
                pools[level] += score
            else:
                pools[level] = score

        # Equal highs/lows por concentración de toques
        highs = window['high'].values
        lows = window['low'].values
        for arr, base_score in ((highs, 3.0), (lows, 3.0)):
            # Mapear cada precio a su bucket y contar ocurrencias
            buckets = {}
            for p in arr:
                b = round(p / binsize)
                buckets[b] = buckets.get(b, 0) + 1
            for b, cnt in buckets.items():
                if cnt >= 2:
                    add(b * binsize, base_score * cnt)

        # Swings (pivotes): más peso porque suelen concentrar stops
        swing_highs = window['max'].dropna().values if 'max' in window.columns else []
        swing_lows = window['min'].dropna().values if 'min' in window.columns else []
        for p in swing_highs:
            add(float(p), 4.0)
        for p in swing_lows:
            add(float(p), 4.0)

        # Bordes de FVG en la ventana
        if 'is_fvg_bearish' in window.columns and 'is_fvg_bullish' in window.columns:
            idxs = window.index
            for j in range(2, len(window)):
                # bullish fvg en vela j-1
                if window['is_fvg_bullish'].iloc[j-1]:
                    # borde que usas en entradas long: low de j (la vela posterior)
                    fvg_border = float(window['low'].iloc[j])
                    add(fvg_border, 2.5)
                if window['is_fvg_bearish'].iloc[j-1]:
                    # borde que usas en entradas short: high de j (la vela posterior)
                    fvg_border = float(window['high'].iloc[j])
                    add(fvg_border, 2.5)

        # Niveles redondos dentro del rango de la ventana
        wmin = float(window['low'].min())
        wmax = float(window['high'].max())
        if step > 0:
            lvl = (np.floor(wmin / step) * step)
            while lvl <= wmax:
                # Cuenta cuántas velas tocaron cerca del nivel
                hits = ((np.abs(window['high'] - lvl) <= binsize) | (np.abs(window['low'] - lvl) <= binsize)).sum()
                if hits >= 1:
                    add(lvl, 0.5 * hits)
                lvl += step

        # Convertir a lista ordenada por score desc
        levels = [{'price': float(k), 'score': float(v)} for k, v in pools.items()]
        levels.sort(key=lambda x: (-x['score'], x['price']))
        return levels

    def select_target_pool(self, i: int, direction: str, entry_price: float, sl_price: float, pools: list):
        """Elige el pool objetivo en la dirección del trade priorizando concentración (score) y cercanía.
        Usa umbral de distancia 1.5*ATR si está disponible.
        """
        if not pools:
            return None
        atr = self.df['atr'].iloc[i] if 'atr' in self.df.columns else np.nan
        max_dist = 1.5 * atr if not np.isnan(atr) and atr is not None else None

        if direction == 'LONG':
            candidates = [p for p in pools if p['price'] > entry_price]
            # ordenar por score desc y distancia asc
            candidates.sort(key=lambda p: (-p['score'], abs(p['price'] - entry_price)))
            if max_dist is not None:
                within = [p for p in candidates if (p['price'] - entry_price) <= max_dist]
                if within:
                    return within[0]['price']
            return candidates[0]['price'] if candidates else None
        else:
            candidates = [p for p in pools if p['price'] < entry_price]
            candidates.sort(key=lambda p: (-p['score'], abs(p['price'] - entry_price)))
            if max_dist is not None:
                within = [p for p in candidates if (entry_price - p['price']) <= max_dist]
                if within:
                    return within[0]['price']
            return candidates[0]['price'] if candidates else None

    def run_backtest(self):
        """Ejecuta el backtest completo."""
        self.df = self.fetch_historical_data()
        if self.df is None: return
        self.find_patterns()
        self.compute_atr()
        self.compute_macd()

        logger.info("Simulando trades (Longs y Shorts)...")
        i = self.structure_lookback + 2
        while i < len(self.df):
            if self.active_trade:
                exit_info = self.manage_active_trade(i)
                if exit_info:
                    i = exit_info['exit_idx'] + 1
                    continue
            
            if self.check_long_setup(i):
                i += 1
                continue

            if self.check_short_setup(i):
                i += 1
                continue
            i += 1

        logger.info(f"=== BACKTEST COMPLETADO PARA {self.symbol} ===")
        self.print_results()
        self.plot_results()
    
    def check_long_setup(self, i):
        recent_lows = self.df['min'].iloc[i-50:i].dropna()
        if len(recent_lows) < 2 or recent_lows.iloc[-1] >= recent_lows.iloc[-2]: return False
        
        sweep_idx = self.df.index.get_loc(recent_lows.index[-1])
        if i - sweep_idx > 12: return False
        
        fvg_window = self.df.iloc[sweep_idx:i]
        bullish_fvgs = fvg_window[fvg_window['is_fvg_bullish']]
        if not bullish_fvgs.empty:
            fvg_candle_idx = self.df.index.get_loc(bullish_fvgs.index[-1])
            fvg_high = self.df['low'].iloc[fvg_candle_idx + 1]
            if self.df['low'].iloc[i] <= fvg_high:
                # Filtro MACD (alcista)
                if self.enable_macd_filter:
                    if 'macd_hist' not in self.df.columns: return False
                    mh = self.df['macd_hist']; m = self.df['macd']; ms = self.df['macd_signal']
                    if i-2 < 0 or np.isnan(mh.iloc[i-1]) or np.isnan(m.iloc[i-1]) or np.isnan(ms.iloc[i-1]):
                        return False
                    if not (mh.iloc[i-1] > 0 and m.iloc[i-1] > ms.iloc[i-1] and mh.iloc[i-1] >= mh.iloc[i-2]):
                        return False
                # Selección de pool objetivo 48h por concentración
                pools = self.build_liquidity_pools(i, lookback=self.pool_lookback_bars, tol=self.equal_tol)
                tp_pool = self.select_target_pool(i, 'LONG', float(fvg_high), float(recent_lows.iloc[-1]), pools)
                self.open_trade(i, 'LONG', float(fvg_high), float(recent_lows.iloc[-1]), tp_override=tp_pool)
                return True
        return False

    def check_short_setup(self, i):
        recent_highs = self.df['max'].iloc[i-50:i].dropna()
        if len(recent_highs) < 2 or recent_highs.iloc[-1] <= recent_highs.iloc[-2]: return False

        sweep_idx = self.df.index.get_loc(recent_highs.index[-1])
        if i - sweep_idx > 12: return False

        fvg_window = self.df.iloc[sweep_idx:i]
        bearish_fvgs = fvg_window[fvg_window['is_fvg_bearish']]
        if not bearish_fvgs.empty:
            fvg_candle_idx = self.df.index.get_loc(bearish_fvgs.index[-1])
            fvg_low = self.df['high'].iloc[fvg_candle_idx + 1]
            if self.df['high'].iloc[i] >= fvg_low:
                # Filtro MACD (bajista)
                if self.enable_macd_filter:
                    if 'macd_hist' not in self.df.columns: return False
                    mh = self.df['macd_hist']; m = self.df['macd']; ms = self.df['macd_signal']
                    if i-2 < 0 or np.isnan(mh.iloc[i-1]) or np.isnan(m.iloc[i-1]) or np.isnan(ms.iloc[i-1]):
                        return False
                    if not (mh.iloc[i-1] < 0 and m.iloc[i-1] < ms.iloc[i-1] and mh.iloc[i-1] <= mh.iloc[i-2]):
                        return False
                pools = self.build_liquidity_pools(i, lookback=self.pool_lookback_bars, tol=self.equal_tol)
                tp_pool = self.select_target_pool(i, 'SHORT', float(fvg_low), float(recent_highs.iloc[-1]), pools)
                self.open_trade(i, 'SHORT', float(fvg_low), float(recent_highs.iloc[-1]), tp_override=tp_pool)
                return True
        return False

    def open_trade(self, entry_idx, direction, entry_price, liquidity_level, tp_override=None):
        """Abre una nueva posición. Si tp_override (pool) respeta R:R mínimo, se usa como TP."""
        if direction == 'LONG':
            # Usar porcentaje configurable para SL
            stop_loss_price = entry_price * (1 - self.sl_percentage / 100)
            risk_per_unit = entry_price - stop_loss_price
            # Usar porcentaje configurable para TP por defecto
            default_tp = entry_price * (1 + self.tp_percentage / 100)
        else:
            # Usar porcentaje configurable para SL
            stop_loss_price = entry_price * (1 + self.sl_percentage / 100)
            risk_per_unit = stop_loss_price - entry_price
            # Usar porcentaje configurable para TP por defecto
            default_tp = entry_price * (1 - self.tp_percentage / 100)

        if risk_per_unit <= 0: return
        # Elegir TP por pool si cumple R:R mínimo
        take_profit_price = default_tp
        if tp_override is not None:
            if direction == 'LONG':
                rr = (tp_override - entry_price) / risk_per_unit
                if rr >= self.min_rr and tp_override > entry_price:
                    take_profit_price = tp_override
            else:
                rr = (entry_price - tp_override) / risk_per_unit
                if rr >= self.min_rr and tp_override < entry_price:
                    take_profit_price = tp_override

        capital_to_risk = self.balance * self.risk_per_trade_pct
        position_size = capital_to_risk / risk_per_unit

        self.active_trade = {
            'entry_idx': entry_idx, 
            'direction': direction, 
            'entry_price': entry_price,
            'sl_price': stop_loss_price,
            'original_sl_price': stop_loss_price, # <<< LÍNEA AÑADIDA
            'tp_price': take_profit_price,
            'position_size': position_size
        }

    def manage_active_trade(self, current_idx):
        """
        Gestiona la salida de un trade activo.
        Prioridad: 1. Take Profit, 2. Stop Loss (Estructural), 3. Time Limit.
        """
        trade = self.active_trade
        
        for i in range(trade['entry_idx'] + 1, current_idx + 1):
            if i >= len(self.df): return None
            
            candle = self.df.iloc[i]
            exit_reason, pnl, exit_price = None, 0, 0

            if trade['direction'] == 'LONG':
                # 1. Actualizar el Trailing Stop Estructural (solo si está habilitado para este símbolo)
                if self.enable_structural_stop:
                    recent_structure_lows = self.df['min'].iloc[trade['entry_idx']:i].dropna()
                    if not recent_structure_lows.empty:
                        new_protective_stop = recent_structure_lows.iloc[-1]
                        if new_protective_stop > trade['sl_price']:
                            trade['sl_price'] = new_protective_stop
                
                # --- LÓGICA DE SALIDA ---
                # 1. Comprobación de Take Profit
                if candle['high'] >= trade['tp_price']:
                    exit_reason, exit_price = 'Take Profit', trade['tp_price']
                
                # 2. Comprobación de Stop Loss
                elif candle['low'] <= trade['sl_price']:
                    exit_price = trade['sl_price']
                    # <<< LÓGICA CORREGIDA (igual que el bot en vivo) >>>
                    exit_reason = 'Stop Loss' if trade['sl_price'] == trade['original_sl_price'] else 'Stop Estructural (Trailing)'

            else:  # SHORT
                # 1. Actualizar el Trailing Stop Estructural (solo si está habilitado para este símbolo)
                if self.enable_structural_stop:
                    recent_structure_highs = self.df['max'].iloc[trade['entry_idx']:i].dropna()
                    if not recent_structure_highs.empty:
                        new_protective_stop = recent_structure_highs.iloc[-1]
                        if new_protective_stop < trade['sl_price']:
                            trade['sl_price'] = new_protective_stop
                
                # --- LÓGICA DE SALIDA ---
                # 1. Comprobación de Take Profit
                if candle['low'] <= trade['tp_price']:
                    exit_reason, exit_price = 'Take Profit', trade['tp_price']
                
                # 2. Comprobación de Stop Loss
                elif candle['high'] >= trade['sl_price']:
                    exit_price = trade['sl_price']
                    # <<< LÓGICA CORREGIDA (igual que el bot en vivo) >>>
                    exit_reason = 'Stop Loss' if trade['sl_price'] == trade['original_sl_price'] else 'Stop Estructural (Trailing)'
            
            # 3. Límite de Tiempo
            if not exit_reason and (i - trade['entry_idx']) >= self.max_candles_in_trade:
                exit_reason, exit_price = 'Time Limit', candle['close']

            if exit_reason:
                pnl = (exit_price - trade['entry_price']) * trade['position_size'] if trade['direction'] == 'LONG' else (trade['entry_price'] - exit_price) * trade['position_size']
                self._log_and_close_trade(trade, i, exit_price, exit_reason, pnl)
                return {'exit_idx': i}
        
        return None
    def _log_and_close_trade(self, trade, exit_idx, exit_price, exit_reason, pnl):
        """Función auxiliar para registrar un trade con todos los detalles."""
        trade_log = {
            'entry_time': self.df.index[trade['entry_idx']],
            'exit_time': self.df.index[exit_idx],
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'stop_loss': trade['sl_price'],
            'original_sl_price': trade.get('original_sl_price', trade['sl_price']),
            'take_profit': trade['tp_price'],
            'pnl': pnl,
            'position_size': trade.get('position_size', None),
            'exit_reason': exit_reason
        }
        self.balance += pnl
        self.trades.append(trade_log)
        self.equity_curve.append({'time': self.df.index[exit_idx], 'balance': self.balance})
        self.active_trade = None
        if callable(getattr(self, 'pnl_callback', None)):
            try:
                self.pnl_callback(self.symbol, pnl, trade_log)
            except Exception as _:
                pass

class MultiSymbolBacktest:
    def __init__(self, symbols: list[str], initial_balance: float = 1000.0, symbol_configs: dict = None):
        self.symbols = [s.replace('USDT', '/USDT') if 'USDT' in s and '/' not in s else s for s in symbols]
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_concurrent = 4
        self.symbol_leverage = {
            'ETH/USDT': 15,
            'HYPE/USDT': 15,
            'SOL/USDT': 15,
        }
        # Configuraciones específicas por símbolo
        self.symbol_configs = symbol_configs or {}
        self.engines: dict[str, SmartMoneyScalpingBacktest] = {}
        self.concurrent_open = 0
        self.combined_equity = [{'time': None, 'balance': self.initial_balance}]
        self.all_trades: list[dict] = []

    def _pnl_sink(self, symbol: str, pnl: float, trade_log: dict):
        self.balance += pnl
        self.all_trades.append({'symbol': symbol, **trade_log})
        self.combined_equity.append({'time': trade_log['exit_time'], 'balance': self.balance})
        self.concurrent_open = sum(1 for e in self.engines.values() if e.active_trade is not None)

    def _fetch_and_prepare(self):
        for s in self.symbols:
            # Obtener configuración específica para este símbolo
            symbol_key = s.replace('/', '')
            symbol_config = self.symbol_configs.get(symbol_key, {})
            
            eng = SmartMoneyScalpingBacktest(symbol=s, initial_balance=self.balance, symbol_config=symbol_config)
            eng.leverage = self.symbol_leverage.get(s, eng.leverage)
            df = eng.fetch_historical_data()
            if df is None or df.empty:
                logger.warning(f"Sin datos para {s}, se omite.")
                continue
            eng.df = df
            eng.find_patterns()
            eng.compute_atr()
            eng.compute_macd()
            eng.pnl_callback = self._pnl_sink
            self.engines[s] = eng
            
            # Log de la configuración aplicada
            logger.info(f"Configuración para {s}: Stop estructural={eng.enable_structural_stop}, TP%={eng.tp_percentage}, SL%={eng.sl_percentage}")

    def run(self):
        self._fetch_and_prepare()
        if not self.engines:
            logger.error("No hay símbolos con datos para backtest multisímbolo.")
            return

        # Timeline global: unión de todos los timestamps
        all_ts = sorted(set().union(*[set(eng.df.index) for eng in self.engines.values()]))

        def loc_idx(eng: SmartMoneyScalpingBacktest, ts):
            try:
                return eng.df.index.get_loc(ts)
            except KeyError:
                return None

        logger.info(f"Multisímbolo: {len(self.engines)} símbolos; {len(all_ts)} timestamps.")

        for ts in all_ts:
            # 1) Gestionar posiciones abiertas
            for s, eng in self.engines.items():
                i = loc_idx(eng, ts)
                if i is None:
                    continue
                if eng.active_trade:
                    eng.manage_active_trade(i)

            # 2) Recalcular concurrencia
            self.concurrent_open = sum(1 for e in self.engines.values() if e.active_trade is not None)

            # 3) Buscar nuevas entradas respetando límite global
            for s, eng in self.engines.items():
                if self.concurrent_open >= self.max_concurrent:
                    break
                i = loc_idx(eng, ts)
                if i is None or i < eng.structure_lookback + 2:
                    continue
                if eng.active_trade:
                    continue

                # Riesgo fijo por señal basado en balance global actual
                eng.balance = self.balance

                opened = False
                long_opened = False
                
                if eng.check_long_setup(i):
                    opened = True
                    long_opened = True
                elif eng.check_short_setup(i):
                    opened = True

                if opened:
                    eng.leverage = self.symbol_leverage.get(s, eng.leverage)
                    self.concurrent_open += 1
                    
                    # FUNCIONALIDAD ESPECIAL: Si ETH abrió un long, también abrir un long en SOL automáticamente
                    # Esto permite aprovechar la correlación entre ETH y SOL en movimientos alcistas
                    if long_opened and s == 'ETH/USDT' and 'SOL/USDT' in self.engines:
                        sol_eng = self.engines['SOL/USDT']
                        sol_i = loc_idx(sol_eng, ts)
                        
                        # Verificar que SOL no tenga una posición activa y que haya datos disponibles
                        if (sol_i is not None and 
                            sol_i >= sol_eng.structure_lookback + 2 and 
                            not sol_eng.active_trade and 
                            self.concurrent_open < self.max_concurrent):
                            
                            # Actualizar balance de SOL y abrir posición long forzada
                            sol_eng.balance = self.balance
                            sol_eng.leverage = self.symbol_leverage.get('SOL/USDT', sol_eng.leverage)
                            
                            # Crear una entrada long forzada en SOL usando el precio actual
                            current_price = float(sol_eng.df['close'].iloc[sol_i])
                            # Usar un stop loss basado en ATR o un porcentaje fijo
                            atr_value = sol_eng.df['atr'].iloc[sol_i] if 'atr' in sol_eng.df.columns and not pd.isna(sol_eng.df['atr'].iloc[sol_i]) else current_price * 0.02
                            stop_loss = current_price - (atr_value * 1.5)
                            
                            # Abrir trade en SOL
                            sol_eng.open_trade(sol_i, 'LONG', current_price, stop_loss)
                            self.concurrent_open += 1
                            
                            logger.info(f"🔗 ETH Long detectado -> Abriendo Long automático en SOL a ${current_price:.4f}")

        self._print_results()
        self._plot_combined_equity()
        self._save_excel_report_multi()

    def _print_results(self):
        print("\n" + "="*60)
        print("RESULTADOS SMC MULTI-SÍMBOLO")
        print("="*60)
        print(f"Balance Final:      ${self.balance:,.2f} | Retorno: {(self.balance/self.initial_balance - 1)*100:,.2f}%")
        total_trades = len(self.all_trades)
        wins = [t for t in self.all_trades if t['pnl'] > 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        print(f"Total de Trades:    {total_trades}")
        print(f"Win Rate:           {win_rate:.2f}%")
        print("="*60)

    def _plot_combined_equity(self):
        eq_df = pd.DataFrame(self.combined_equity).dropna().set_index('time')
        if eq_df.empty:
            return
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(eq_df.index, eq_df['balance'], label='Balance', color='#0077b6', linewidth=1.5)
        ax.fill_between(eq_df.index, self.initial_balance, eq_df['balance'], where=(eq_df['balance'] >= self.initial_balance), color='#2ca02c', alpha=0.3, interpolate=True)
        ax.set_title('Curva de Equity - SMC Multisímbolo', fontsize=16, fontweight='bold')
        filename = 'equity_curve_SMC_multisymbol.png'
        plt.savefig(filename, dpi=300)
        logger.info(f"Gráfico combinado guardado como: {filename}")
        plt.show()

    def _save_excel_report_multi(self):
        trades_df = pd.DataFrame(self.all_trades)
        filename = f"report_SMC_MULTI_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        logger.info(f"Generando reporte Excel multisímbolo: {filename}")

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1, 'align': 'center'})
            money_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})

            # Resumen General
            summary_sheet = workbook.add_worksheet('Resumen General')

            total_trades = len(trades_df)
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] <= 0]
            win_rate = (len(wins) / total_trades) if total_trades > 0 else 0

            eq_df = pd.DataFrame(self.combined_equity).dropna().set_index('time')
            if not eq_df.empty:
                eq_df['peak'] = eq_df['balance'].cummax()
                eq_df['drawdown_pct'] = (eq_df['balance'] - eq_df['peak']) / eq_df['peak']
                max_drawdown = eq_df['drawdown_pct'].min()
            else:
                max_drawdown = 0

            # Calcular Profit Factor de forma segura
            profit_factor = 0
            if not losses.empty and losses['pnl'].sum() != 0:
                profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum())
            elif not wins.empty and losses.empty:
                profit_factor = 999.99  # Valor alto pero finito cuando no hay pérdidas
            
            summary_data = {
                "Balance Inicial": self.initial_balance,
                "Balance Final": self.balance,
                "Retorno Total": (self.balance / self.initial_balance) - 1,
                "P/L Neto": trades_df['pnl'].sum() if not trades_df.empty else 0,
                "Max Drawdown": max_drawdown if not np.isnan(max_drawdown) and not np.isinf(max_drawdown) else 0,
                "Total de Trades": total_trades,
                "Win Rate": win_rate if not np.isnan(win_rate) and not np.isinf(win_rate) else 0,
                "Profit Factor": profit_factor if not np.isnan(profit_factor) and not np.isinf(profit_factor) else 0,
                "Ganancia Promedio": wins['pnl'].mean() if not wins.empty and not np.isnan(wins['pnl'].mean()) else 0,
                "Pérdida Promedio": losses['pnl'].mean() if not losses.empty and not np.isnan(losses['pnl'].mean()) else 0
            }

            summary_sheet.write_row('A1', ['Métrica', 'Valor'], header_format)
            row = 1
            for key, value in summary_data.items():
                summary_sheet.write(row, 0, key)
                fmt = money_format
                if "Retorno" in key or "Rate" in key or "Drawdown" in key:
                    fmt = percent_format
                elif "Trades" in key or "Factor" in key:
                    fmt = None
                summary_sheet.write(row, 1, value, fmt)
                row += 1
            summary_sheet.set_column('A:A', 25); summary_sheet.set_column('B:B', 18)

            # Todos los Trades (con símbolo)
            if not trades_df.empty:
                trades_df_excel = trades_df.copy()
                if 'entry_time' in trades_df_excel.columns:
                    trades_df_excel['entry_time'] = pd.to_datetime(trades_df_excel['entry_time']).dt.tz_localize(None)
                if 'exit_time' in trades_df_excel.columns:
                    trades_df_excel['exit_time'] = pd.to_datetime(trades_df_excel['exit_time']).dt.tz_localize(None)
                trades_df_excel.to_excel(writer, sheet_name='Todos los Trades', index=False)

                # Desempeño por Días
                daily_sheet = workbook.add_worksheet('Desempeño por Días')
                daily_perf = trades_df.copy()
                daily_perf['date'] = pd.to_datetime(daily_perf['entry_time']).dt.tz_localize(None).dt.date
                daily_agg = daily_perf.groupby('date').agg(total_pnl=('pnl', 'sum')).reset_index()
                daily_sheet.write_row('A1', ['Fecha', 'P/L Total'], header_format)
                for r_idx, r in daily_agg.iterrows():
                    daily_sheet.write(r_idx + 1, 0, r['date'].strftime('%Y-%m-%d'))
                    daily_sheet.write(r_idx + 1, 1, r['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'})
                chart.add_series({'name': 'P/L Diario', 'categories': f"='Desempeño por Días'!$A$2:$A${len(daily_agg)+1}", 'values': f"='Desempeño por Días'!$B$2:$B${len(daily_agg)+1}"})
                daily_sheet.insert_chart('D2', chart)

                # Desempeño por Horas
                hourly_sheet = workbook.add_worksheet('Desempeño por Horas')
                hourly_perf = trades_df.copy()
                hourly_perf['hour'] = pd.to_datetime(hourly_perf['entry_time']).dt.tz_localize(None).dt.hour
                hourly_agg = hourly_perf.groupby('hour').agg(total_pnl=('pnl', 'sum')).reset_index()
                hourly_sheet.write_row('A1', ['Hora', 'P/L Total'], header_format)
                for r_idx, r in hourly_agg.iterrows():
                    hourly_sheet.write(r_idx + 1, 0, r['hour'])
                    hourly_sheet.write(r_idx + 1, 1, r['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'})
                chart.add_series({'name': 'P/L por Hora', 'categories': f"='Desempeño por Horas'!$A$2:$A${len(hourly_agg)+1}", 'values': f"='Desempeño por Horas'!$B$2:$B${len(hourly_agg)+1}"})
                hourly_sheet.insert_chart('D2', chart)

                # Desempeño por Día Semana
                weekday_sheet = workbook.add_worksheet('Desempeño por Día Semana')
                weekday_perf = trades_df.copy()
                weekday_perf['weekday'] = pd.to_datetime(weekday_perf['entry_time']).dt.tz_localize(None).dt.weekday
                day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                weekday_agg = weekday_perf.groupby('weekday').agg(total_pnl=('pnl', 'sum')).reindex(range(7)).fillna(0).reset_index()
                weekday_agg['weekday'] = weekday_agg['weekday'].map(lambda x: day_names[x])
                weekday_sheet.write_row('A1', ['Día', 'P/L Total'], header_format)
                for r_idx, r in weekday_agg.iterrows():
                    weekday_sheet.write(r_idx + 1, 0, r['weekday'])
                    weekday_sheet.write(r_idx + 1, 1, r['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'})
                chart.add_series({'name': 'P/L por Día', 'categories': "='Desempeño por Día Semana'!$A$2:$A$8", 'values': "='Desempeño por Día Semana'!$B$2:$B$8"})
                weekday_sheet.insert_chart('D2', chart)

            # Resumen por Símbolo + Trades por Símbolo
            if not trades_df.empty and 'symbol' in trades_df.columns:
                # Calcular R:R realizado cuando hay datos suficientes
                df_rr = trades_df.copy()
                def compute_rr(row):
                    try:
                        pos_size = float(row.get('position_size')) if row.get('position_size') is not None else None
                        orig_sl = float(row.get('original_sl_price')) if row.get('original_sl_price') is not None else None
                        entry = float(row['entry_price'])
                        pnl_total = float(row['pnl'])
                        if pos_size is None or pos_size == 0 or orig_sl is None:
                            return np.nan
                        per_unit_pnl = pnl_total / pos_size
                        if row['direction'] == 'LONG':
                            risk_per_unit = entry - orig_sl
                        else:
                            risk_per_unit = orig_sl - entry
                        if risk_per_unit <= 0:
                            return np.nan
                        return per_unit_pnl / risk_per_unit
                    except Exception:
                        return np.nan
                df_rr['rr'] = df_rr.apply(compute_rr, axis=1)

                # Hoja: Resumen por Símbolo
                sym_summary_sheet = workbook.add_worksheet('Resumen por Símbolo')
                sym_summary_sheet.write_row('A1', ['Símbolo', 'P/L Neto', 'Trades', 'Win Rate', 'Profit Factor', 'Avg Win', 'Avg Loss', 'RR Medio', 'RR Mediano'], header_format)
                r = 1
                for sym, df_sym in df_rr.groupby('symbol'):
                    wins_sym = df_sym[df_sym['pnl'] > 0]
                    losses_sym = df_sym[df_sym['pnl'] <= 0]
                    total_sym = len(df_sym)
                    win_rate_sym = (len(wins_sym) / total_sym) if total_sym > 0 else 0
                    # Calcular Profit Factor de forma segura por símbolo
                    if not losses_sym.empty and losses_sym['pnl'].sum() != 0:
                        pf_sym = wins_sym['pnl'].sum() / abs(losses_sym['pnl'].sum())
                    elif not wins_sym.empty and losses_sym.empty:
                        pf_sym = 999.99  # Valor alto pero finito
                    else:
                        pf_sym = 0
                    avg_win = wins_sym['pnl'].mean() if not wins_sym.empty else 0
                    avg_loss = losses_sym['pnl'].mean() if not losses_sym.empty else 0
                    sym_summary_sheet.write(r, 0, sym)
                    sym_summary_sheet.write(r, 1, df_sym['pnl'].sum(), money_format)
                    sym_summary_sheet.write(r, 2, total_sym)
                    sym_summary_sheet.write(r, 3, win_rate_sym, percent_format)
                    sym_summary_sheet.write(r, 4, pf_sym)
                    sym_summary_sheet.write(r, 5, avg_win, money_format)
                    sym_summary_sheet.write(r, 6, avg_loss, money_format)
                    # Validar valores RR para evitar NaN
                    rr_mean = df_sym['rr'].mean() if 'rr' in df_sym.columns and not df_sym['rr'].empty else 0
                    rr_median = df_sym['rr'].median() if 'rr' in df_sym.columns and not df_sym['rr'].empty else 0
                    
                    # Asegurar que no sean NaN o INF
                    rr_mean = rr_mean if not np.isnan(rr_mean) and not np.isinf(rr_mean) else 0
                    rr_median = rr_median if not np.isnan(rr_median) and not np.isinf(rr_median) else 0
                    
                    sym_summary_sheet.write(r, 7, rr_mean)
                    sym_summary_sheet.write(r, 8, rr_median)
                    r += 1
                sym_summary_sheet.set_column('A:A', 14); sym_summary_sheet.set_column('B:I', 14)

                # Hojas: Trades por símbolo
                for sym, df_sym in df_rr.groupby('symbol'):
                    sheet_name = f"Trades_{str(sym).replace('/', '_')}"
                    # Evitar nombres duplicados o largos
                    sheet_name = sheet_name[:31]
                    ws = workbook.add_worksheet(sheet_name)
                    df_x = df_sym.copy()
                    if 'entry_time' in df_x.columns:
                        df_x['entry_time'] = pd.to_datetime(df_x['entry_time']).dt.tz_localize(None)
                    if 'exit_time' in df_x.columns:
                        df_x['exit_time'] = pd.to_datetime(df_x['exit_time']).dt.tz_localize(None)
                    # Escribir encabezados
                    headers = list(df_x.columns)
                    ws.write_row('A1', headers, header_format)
                    # Escribir filas
                    for ridx, row in enumerate(df_x.itertuples(index=False), start=2):
                        for cidx, val in enumerate(row, start=1):
                            ws.write(ridx-1, cidx-1, val)

                # Métricas por Timeframe de Entrada / Salida
                # Entradas por hora
                met_ent = df_rr.copy()
                met_ent['entry_hour'] = pd.to_datetime(met_ent['entry_time']).dt.tz_localize(None).dt.hour
                ent_group = met_ent.groupby('entry_hour').agg(
                    trades=('pnl', 'count'),
                    win_rate=('pnl', lambda s: (s > 0).mean()),
                    avg_pnl=('pnl', 'mean'),
                    rr_mean=('rr', 'mean')
                ).reset_index().sort_values('entry_hour')
                ent_sheet = workbook.add_worksheet('Metricas Entradas (Hora)')
                ent_sheet.write_row('A1', ['Hora Entrada', 'Trades', 'Win Rate', 'Avg PnL', 'RR Medio'], header_format)
                for ridx, rrow in ent_group.iterrows():
                    ent_sheet.write(ridx + 1, 0, int(rrow['entry_hour']))
                    ent_sheet.write(ridx + 1, 1, int(rrow['trades']))
                    ent_sheet.write(ridx + 1, 2, float(rrow['win_rate']) if not np.isnan(rrow['win_rate']) else 0, percent_format)
                    ent_sheet.write(ridx + 1, 3, float(rrow['avg_pnl']) if not np.isnan(rrow['avg_pnl']) else 0, money_format)
                    ent_sheet.write(ridx + 1, 4, float(rrow['rr_mean']) if not np.isnan(rrow['rr_mean']) else 0)

                # Salidas por hora
                met_sal = df_rr.copy()
                met_sal['exit_hour'] = pd.to_datetime(met_sal['exit_time']).dt.tz_localize(None).dt.hour
                sal_group = met_sal.groupby('exit_hour').agg(
                    trades=('pnl', 'count'),
                    win_rate=('pnl', lambda s: (s > 0).mean()),
                    avg_pnl=('pnl', 'mean'),
                    rr_mean=('rr', 'mean')
                ).reset_index().sort_values('exit_hour')
                sal_sheet = workbook.add_worksheet('Metricas Salidas (Hora)')
                sal_sheet.write_row('A1', ['Hora Salida', 'Trades', 'Win Rate', 'Avg PnL', 'RR Medio'], header_format)
                for ridx, rrow in sal_group.iterrows():
                    sal_sheet.write(ridx + 1, 0, int(rrow['exit_hour']))
                    sal_sheet.write(ridx + 1, 1, int(rrow['trades']))
                    sal_sheet.write(ridx + 1, 2, float(rrow['win_rate']) if not np.isnan(rrow['win_rate']) else 0, percent_format)
                    sal_sheet.write(ridx + 1, 3, float(rrow['avg_pnl']) if not np.isnan(rrow['avg_pnl']) else 0, money_format)
                    sal_sheet.write(ridx + 1, 4, float(rrow['rr_mean']) if not np.isnan(rrow['rr_mean']) else 0)

                # RR por Símbolo (tabla de resumen percentiles)
                rr_sheet = workbook.add_worksheet('RR por Símbolo')
                rr_sheet.write_row('A1', ['Símbolo', 'Count', 'RR Medio', 'RR Mediano', 'P25', 'P75', '>=1.5 RR %'], header_format)
                r = 1
                for sym, df_sym in df_rr.groupby('symbol'):
                    rr_series = df_sym['rr'].dropna()
                    if rr_series.empty:
                        rr_sheet.write(r, 0, sym)
                        rr_sheet.write(r, 1, 0)
                        rr_sheet.write(r, 2, 0)
                        rr_sheet.write(r, 3, 0)
                        rr_sheet.write(r, 4, 0)
                        rr_sheet.write(r, 5, 0)
                        rr_sheet.write(r, 6, 0, percent_format)
                        r += 1
                        continue
                    
                    # Validar todos los valores para evitar NaN/INF
                    p25 = float(rr_series.quantile(0.25))
                    p75 = float(rr_series.quantile(0.75))
                    over_15 = float((rr_series >= 1.5).mean())
                    rr_mean = float(rr_series.mean())
                    rr_median = float(rr_series.median())
                    
                    # Asegurar que no sean NaN o INF
                    p25 = p25 if not np.isnan(p25) and not np.isinf(p25) else 0
                    p75 = p75 if not np.isnan(p75) and not np.isinf(p75) else 0
                    over_15 = over_15 if not np.isnan(over_15) and not np.isinf(over_15) else 0
                    rr_mean = rr_mean if not np.isnan(rr_mean) and not np.isinf(rr_mean) else 0
                    rr_median = rr_median if not np.isnan(rr_median) and not np.isinf(rr_median) else 0
                    
                    rr_sheet.write(r, 0, sym)
                    rr_sheet.write(r, 1, int(rr_series.count()))
                    rr_sheet.write(r, 2, rr_mean)
                    rr_sheet.write(r, 3, rr_median)
                    rr_sheet.write(r, 4, p25)
                    rr_sheet.write(r, 5, p75)
                    rr_sheet.write(r, 6, over_15, percent_format)
                    r += 1

        logger.info("✅ Reporte Excel multisímbolo guardado.")

    def plot_results(self):
        """Grafica la curva de equity."""
        if len(self.equity_curve) <= 1: return
        
        equity_df = pd.DataFrame(self.equity_curve).dropna().set_index('time')
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(equity_df.index, equity_df['balance'], label='Balance', color='#0077b6', linewidth=1.5)
        ax.fill_between(equity_df.index, self.initial_balance, equity_df['balance'], where=(equity_df['balance'] >= self.initial_balance), color='#2ca02c', alpha=0.3, interpolate=True)
        ax.set_title(f'Curva de Equity - SMC Bidireccional - {self.symbol}', fontsize=16, fontweight='bold')
        filename = f"equity_curve_SMC_bidirectional_{self.symbol.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300)
        logger.info(f"Gráfico guardado como: {filename}")
        plt.show()

    def save_excel_report(self, trades_df: pd.DataFrame):
        """Guarda un reporte completo en Excel con múltiples hojas y gráficos."""
        filename = f"report_SMC_{self.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        logger.info(f"Generando reporte Excel: {filename}")
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1, 'align': 'center'})
            money_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # --- Hoja 1: Resumen General ---
            summary_sheet = workbook.add_worksheet('Resumen General')
            
            total_trades = len(trades_df)
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] <= 0]
            win_rate = (len(wins) / total_trades) if total_trades > 0 else 0
            
            equity_df = pd.DataFrame(self.equity_curve).dropna().set_index('time')
            if not equity_df.empty:
                equity_df['peak'] = equity_df['balance'].cummax()
                equity_df['drawdown_pct'] = (equity_df['balance'] - equity_df['peak']) / equity_df['peak']
                max_drawdown = equity_df['drawdown_pct'].min()
            else: max_drawdown = 0

            summary_data = {
                "Balance Inicial": self.initial_balance, "Balance Final": self.balance,
                "Retorno Total": (self.balance / self.initial_balance) - 1, "P/L Neto": trades_df['pnl'].sum(),
                "Max Drawdown": max_drawdown, "Total de Trades": total_trades, "Win Rate": win_rate,
                "Profit Factor": wins['pnl'].sum() / abs(losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf'),
                "Ganancia Promedio": wins['pnl'].mean() if len(wins) > 0 else 0,
                "Pérdida Promedio": losses['pnl'].mean() if len(losses) > 0 else 0
            }

            summary_sheet.write_row('A1', ['Métrica', 'Valor'], header_format)
            row = 1
            for key, value in summary_data.items():
                summary_sheet.write(row, 0, key)
                fmt = money_format
                if "Retorno" in key or "Rate" in key or "Drawdown" in key: fmt = percent_format
                elif "Trades" in key or "Factor" in key: fmt = None
                summary_sheet.write(row, 1, value, fmt)
                row += 1
            summary_sheet.set_column('A:A', 20); summary_sheet.set_column('B:B', 15)

            # --- Hoja 2: Todos los Trades ---
            # ## <<< CORRECCIÓN CLAVE PARA EL ERROR DE ZONA HORARIA >>>
            trades_df_excel = trades_df.copy()
            trades_df_excel['entry_time'] = trades_df_excel['entry_time'].dt.tz_localize(None)
            trades_df_excel['exit_time'] = trades_df_excel['exit_time'].dt.tz_localize(None)
            trades_df_excel.to_excel(writer, sheet_name='Todos los Trades', index=False)
            
            # --- Lógica para las demás hojas ---
            if not trades_df.empty:
                # Daily
                daily_sheet = workbook.add_worksheet('Desempeño por Días')
                daily_perf = trades_df.groupby(trades_df['entry_time'].dt.date).agg(total_pnl=('pnl', 'sum')).reset_index()
                daily_sheet.write_row('A1', ['Fecha', 'P/L Total'], header_format)
                for r_idx, row in daily_perf.iterrows():
                    daily_sheet.write(r_idx + 1, 0, row['entry_time'].strftime('%Y-%m-%d')); daily_sheet.write(r_idx + 1, 1, row['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'}); chart.add_series({'name': 'P/L Diario', 'categories': "='Desempeño por Días'!$A$2:$A$" + str(len(daily_perf)+1), 'values': "='Desempeño por Días'!$B$2:$B$" + str(len(daily_perf)+1)}); daily_sheet.insert_chart('D2', chart)

                # Hourly
                hourly_sheet = workbook.add_worksheet('Desempeño por Horas')
                hourly_perf = trades_df.groupby(trades_df['entry_time'].dt.hour).agg(total_pnl=('pnl', 'sum')).reset_index()
                hourly_sheet.write_row('A1', ['Hora', 'P/L Total'], header_format)
                for r_idx, row in hourly_perf.iterrows():
                    hourly_sheet.write(r_idx + 1, 0, row['entry_time']); hourly_sheet.write(r_idx + 1, 1, row['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'}); chart.add_series({'name': 'P/L por Hora', 'categories': "='Desempeño por Horas'!$A$2:$A$" + str(len(hourly_perf)+1), 'values': "='Desempeño por Horas'!$B$2:$B$" + str(len(hourly_perf)+1)}); hourly_sheet.insert_chart('D2', chart)

                # Weekday
                weekday_sheet = workbook.add_worksheet('Desempeño por Día Semana')
                day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                trades_df['weekday'] = trades_df['entry_time'].dt.weekday
                weekday_perf = trades_df.groupby('weekday').agg(total_pnl=('pnl', 'sum')).reindex(range(7)).fillna(0).reset_index()
                weekday_perf['weekday'] = weekday_perf['weekday'].map(lambda x: day_names[x])
                weekday_sheet.write_row('A1', ['Día', 'P/L Total'], header_format)
                for r_idx, row in weekday_perf.iterrows():
                    weekday_sheet.write(r_idx + 1, 0, row['weekday']); weekday_sheet.write(r_idx + 1, 1, row['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'}); chart.add_series({'name': 'P/L por Día', 'categories': "='Desempeño por Día Semana'!$A$2:$A$8", 'values': "='Desempeño por Día Semana'!$B$2:$B$8"}); weekday_sheet.insert_chart('D2', chart)

        logger.info("✅ Reporte Excel guardado.")

if __name__ == "__main__":
    # Cargar configuración para multisímbolo
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_filename = 'cofigETHBTC.json'
    config_path = os.path.join(script_dir, config_filename)

    symbols_cfg = ['ETHUSDT', 'SOLUSDT', 'HYPEUSDT']
    initial_balance_cfg = 30.0
    symbol_configs_cfg = {}
    max_concurrent_cfg = 4
    
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            symbols_cfg = cfg.get('symbols', symbols_cfg)
            initial_balance_cfg = float(cfg.get('initial_balance', initial_balance_cfg))
            symbol_configs_cfg = cfg.get('symbol_configs', {})
            max_concurrent_cfg = cfg.get('max_concurrent_open', max_concurrent_cfg)
            
        logger.info(f"Configuración cargada desde {config_filename}")
        logger.info(f"Símbolos: {symbols_cfg}")
        logger.info(f"Balance inicial: ${initial_balance_cfg}")
        logger.info(f"Configuraciones por símbolo: {len(symbol_configs_cfg)} símbolos configurados")
        
    except FileNotFoundError:
        logger.warning(f"No se encontró '{config_filename}'. Usando configuración por defecto: {symbols_cfg}")
    except Exception as e:
        logger.error(f"Error leyendo '{config_filename}': {e}")

    msb = MultiSymbolBacktest(symbols=symbols_cfg, initial_balance=initial_balance_cfg, symbol_configs=symbol_configs_cfg)
    msb.max_concurrent = max_concurrent_cfg
    msb.run()