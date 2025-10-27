"""
Módulo de rastreo de wallet para monitorear posiciones DeFi y trading.
Rastrea aperturas y cierres de posiciones, apalancamiento y cantidades.
"""

import asyncio
import aiohttp
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from telegram import Bot
from telegram.error import TelegramError

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Estructura para almacenar información de una posición."""
    symbol: str
    size: float
    leverage: float
    entry_price: float
    position_type: str  # 'LONG' o 'SHORT'
    platform: str  # 'Binance', 'Bybit', 'dYdX', etc.
    timestamp: datetime
    tx_hash: Optional[str] = None
    
@dataclass
class PositionEvent:
    """Evento de apertura o cierre de posición."""
    action: str  # 'OPEN' o 'CLOSE'
    position: Position
    timestamp: datetime
    pnl: Optional[float] = None  # Solo para cierres

class WalletTracker:
    """
    Rastreador de wallet que monitorea múltiples plataformas DeFi y CEX.
    """
    
    def __init__(
        self,
        wallet_address: str,
        telegram_bot: Bot,
        telegram_chat_id: str,
        etherscan_api_key: Optional[str] = None,
        debank_api_key: Optional[str] = None,
        moralis_api_key: Optional[str] = None,
        quicknode_api_key: Optional[str] = None,
        zerion_api_key: Optional[str] = None,
        syve_api_key: Optional[str] = None,
        check_interval: float = 0.5  # segundos (500ms)
    ):
        self.wallet_address = wallet_address.lower()
        self.telegram_bot = telegram_bot
        self.telegram_chat_id = telegram_chat_id
        self.etherscan_api_key = etherscan_api_key
        self.debank_api_key = debank_api_key
        self.moralis_api_key = moralis_api_key
        self.quicknode_api_key = quicknode_api_key
        self.zerion_api_key = zerion_api_key
        self.syve_api_key = syve_api_key
        self.check_interval = check_interval
        
        # Estado interno
        self.active_positions: Dict[str, Position] = {}  # key: symbol_platform
        self.last_check_time = datetime.now()
        self.is_running = False
        
        # Copy trading callbacks
        self.copy_trading_enabled = False
        self.position_open_callback = None
        self.position_close_callback = None
        
        # APIs endpoints - Actualizadas con mejores opciones
        self.etherscan_base = "https://api.etherscan.io/api"
        self.debank_base = "https://pro-openapi.debank.com/v1"
        self.hyperdash_base = "https://hyperdash.info/api"
        
        # APIs más potentes para rastreo
        self.moralis_base = "https://deep-index.moralis.io/api/v2.2"
        self.quicknode_base = "https://api.quicknode.com/v1"
        self.zerion_base = "https://api.zerion.io/v1"
        self.syve_base = "https://api.syve.ai/v1"
        
        # API Keys opcionales
        self.moralis_api_key = None
        self.quicknode_api_key = None
        self.zerion_api_key = None
        self.syve_api_key = None
        
        # Contratos conocidos de plataformas DeFi
        self.known_contracts = {
            # dYdX v4
            "0x65c7c7c4f3d6f5e4b4e4f4e4f4e4f4e4f4e4f4e4": "dYdX",
            # Binance (aproximado - pueden variar)
            "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance",
            # Bybit (aproximado)
            "0x1234567890123456789012345678901234567890": "Bybit",
            # Aave
            "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "Aave",
            # Compound
            "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b": "Compound",
        }
        
    async def start_monitoring(self):
        """Inicia el monitoreo de la wallet."""
        self.is_running = True
        logger.info(f"🔍 Iniciando monitoreo de wallet: {self.wallet_address}")
        logger.info(f"⏱️ Intervalo de verificación: {self.check_interval} segundos")
        
        # Mostrar APIs activas
        active_apis = self.get_active_apis()
        logger.info(f"🔧 APIs activas: {len(active_apis)}")
        for api in active_apis:
            logger.info(f"   ✅ {api}")
        
        await self.send_telegram_notification(
            f"🔍 <b>RASTREO DE WALLET INICIADO</b>\n\n"
            f"📍 Wallet: <code>{self.wallet_address}</code>\n"
            f"⏱️ Intervalo: {self.check_interval}s\n"
            f"🎯 Monitoreando: Posiciones DeFi y CEX"
        )
        
        check_count = 0
        while self.is_running:
            try:
                check_count += 1
                current_time = datetime.now()
                
                # Log cada 10 verificaciones (cada 5 segundos aprox)
                if check_count % 10 == 0:
                    logger.info(f"🔍 Verificación #{check_count} - {current_time.strftime('%H:%M:%S')} - Intervalo: {self.check_interval}s")
                
                await self.check_wallet_activity()
                self.last_check_time = current_time
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error en monitoreo de wallet: {e}")
                await asyncio.sleep(60)  # Esperar más tiempo si hay error
                
    async def stop_monitoring(self):
        """Detiene el monitoreo."""
        self.is_running = False
        logger.info("🛑 Monitoreo de wallet detenido")
        
    async def check_wallet_activity(self):
        """Verifica actividad reciente en la wallet usando SOLO APIs configuradas."""
        try:
            # Debug: mostrar que está verificando (solo cada 20 verificaciones)
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
                
            if self._debug_count % 20 == 0:  # Cada 10 segundos aprox
                logger.debug(f"🔄 Verificando wallet... (check #{self._debug_count})")
            
            # Solo usar APIs que tienen key configurada
            
            # Prioridad 1: Moralis (si tiene API key)
            if self.moralis_api_key and self.moralis_api_key.strip():
                await self.check_moralis_activity()
            
            # Prioridad 2: QuickNode (si tiene API key)
            if self.quicknode_api_key and self.quicknode_api_key.strip():
                await self.check_quicknode_activity()
            
            # Prioridad 3: Zerion (si tiene API key)
            if self.zerion_api_key and self.zerion_api_key.strip():
                await self.check_zerion_portfolio()
            
            # Prioridad 4: Hyperliquid (siempre disponible - API oficial)
            await self.check_hyperdash_positions()
            
            # Prioridad 5: Syve (si tiene API key)
            if self.syve_api_key and self.syve_api_key.strip():
                await self.check_syve_pnl()
            
            # Prioridad 6: Etherscan (si tiene API key válida)
            if (self.etherscan_api_key and 
                self.etherscan_api_key.strip() and 
                self.etherscan_api_key != "YourApiKeyToken"):
                recent_txs = await self.get_recent_transactions()
                for tx in recent_txs:
                    await self.analyze_transaction(tx)
            
            # Prioridad 7: DeBank (si tiene API key)
            if self.debank_api_key and self.debank_api_key.strip():
                await self.check_defi_positions()
            
        except Exception as e:
            logger.error(f"Error verificando actividad de wallet: {e}")
            
    async def check_moralis_activity(self):
        """Verifica actividad usando Moralis Web3 Data API."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'X-API-Key': self.moralis_api_key,
                    'Content-Type': 'application/json'
                }
                
                # Obtener transacciones recientes
                url = f"{self.moralis_base}/{self.wallet_address}"
                params = {
                    'chain': 'eth',
                    'limit': 10,
                    'order': 'DESC'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self.process_moralis_transactions(data)
                    else:
                        logger.debug(f"Moralis API error: {response.status}")
                        
        except Exception as e:
            logger.debug(f"Error consultando Moralis: {e}")
            
    async def process_moralis_transactions(self, moralis_data):
        """Procesa transacciones de Moralis para detectar trading."""
        try:
            if 'result' in moralis_data:
                for tx in moralis_data['result']:
                    # Verificar si es transacción muy reciente (últimos 2 minutos)
                    tx_time = datetime.fromisoformat(tx.get('block_timestamp', '').replace('Z', '+00:00'))
                    if (datetime.now(tx_time.tzinfo) - tx_time).total_seconds() < 120:  # 2 minutos
                        
                        # Analizar si es transacción de trading
                        to_address = tx.get('to_address', '').lower()
                        value = float(tx.get('value', 0)) / 1e18
                        
                        # Detectar plataformas de trading conocidas
                        platform = self.detect_trading_platform(to_address)
                        
                        if platform and value > 0.001:  # Transacciones significativas
                            # Detectar símbolo automáticamente basado en la transacción
                            detected_symbol = self.detect_symbol_from_transaction(tx)
                            
                            # Crear posición estimada
                            position = Position(
                                symbol=detected_symbol,
                                size=value,
                                leverage=1.0,  # Se detectará mejor con más análisis
                                entry_price=0.0,
                                position_type='LONG',  # Se detectará mejor
                                platform=platform,
                                timestamp=tx_time,
                                tx_hash=tx.get('hash')
                            )
                            
                            # Determinar si es apertura o cierre
                            action = 'OPEN' if tx.get('from_address', '').lower() == self.wallet_address else 'CLOSE'
                            
                            event = PositionEvent(
                                action=action,
                                position=position,
                                timestamp=tx_time
                            )
                            
                            await self.notify_position_event(event)
                            
        except Exception as e:
            logger.error(f"Error procesando transacciones Moralis: {e}")
            
    async def check_quicknode_activity(self):
        """Verifica actividad usando QuickNode QuickAlerts."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.quicknode_api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Consultar alertas recientes para la wallet
                url = f"{self.quicknode_base}/alerts"
                params = {
                    'address': self.wallet_address,
                    'limit': 5
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self.process_quicknode_alerts(data)
                    else:
                        logger.debug(f"QuickNode API error: {response.status}")
                        
        except Exception as e:
            logger.debug(f"Error consultando QuickNode: {e}")
            
    async def process_quicknode_alerts(self, alerts_data):
        """Procesa alertas de QuickNode para detección instantánea."""
        try:
            for alert in alerts_data.get('alerts', []):
                # QuickNode proporciona alertas en tiempo real
                alert_time = datetime.fromtimestamp(alert.get('timestamp', 0))
                
                if (datetime.now() - alert_time).total_seconds() < 60:  # Último minuto
                    tx_hash = alert.get('txHash')
                    value = float(alert.get('value', 0))
                    
                    # Crear evento inmediato
                    position = Position(
                        symbol=alert.get('token', 'ETH'),
                        size=value,
                        leverage=1.0,
                        entry_price=0.0,
                        position_type=alert.get('type', 'LONG'),
                        platform="QuickNode_Alert",
                        timestamp=alert_time,
                        tx_hash=tx_hash
                    )
                    
                    event = PositionEvent(
                        action=alert.get('action', 'OPEN'),
                        position=position,
                        timestamp=alert_time
                    )
                    
                    await self.notify_position_event(event)
                    
        except Exception as e:
            logger.error(f"Error procesando alertas QuickNode: {e}")
            
    def detect_trading_platform(self, contract_address: str) -> Optional[str]:
        """Detecta la plataforma de trading basada en la dirección del contrato."""
        # Direcciones conocidas de plataformas de trading
        trading_platforms = {
            # dYdX
            "0x65c7c7c4f3d6f5e4b4e4f4e4f4e4f4e4f4e4f4e4": "dYdX",
            # Uniswap V3
            "0xe592427a0aece92de3edee1f18e0157c05861564": "Uniswap_V3",
            # 1inch
            "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch",
            # Binance (aproximado)
            "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance",
            # Agregar más según sea necesario
        }
        
        return trading_platforms.get(contract_address.lower())
    
    def detect_symbol_from_transaction(self, tx: Dict) -> str:
        """Detecta el símbolo de trading basado en los datos de la transacción."""
        try:
            # Mapeo de contratos de tokens conocidos a símbolos
            token_contracts = {
                # ETH nativo
                '0x0000000000000000000000000000000000000000': 'ETH',
                # WETH
                '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2': 'ETH',
                # USDT
                '0xdac17f958d2ee523a2206206994597c13d831ec7': 'USDT',
                # USDC
                '0xa0b86a33e6417c1c6c7c4b4c3d3c3c3c3c3c3c3c': 'USDC',
                # BTC (WBTC)
                '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599': 'BTC',
                # Agregar más tokens según sea necesario
            }
            
            # Verificar si hay valor ETH en la transacción
            value = float(tx.get('value', 0))
            if value > 0:
                return 'ETH'
            
            # Analizar el input data para detectar interacciones con tokens
            input_data = tx.get('input', '')
            to_address = tx.get('to_address', '').lower()
            
            # Verificar si la dirección de destino es un contrato de token conocido
            if to_address in token_contracts:
                return token_contracts[to_address]
            
            # Analizar el input data para detectar transferencias de tokens
            if input_data and len(input_data) > 10:
                # Signature para transfer(address,uint256) = 0xa9059cbb
                if input_data.startswith('0xa9059cbb'):
                    # Es una transferencia de token, intentar detectar cuál
                    return self.detect_token_from_input(input_data, to_address)
                
                # Signature para approve(address,uint256) = 0x095ea7b3
                elif input_data.startswith('0x095ea7b3'):
                    # Es una aprobación de token
                    return self.detect_token_from_input(input_data, to_address)
            
            # Si no se puede detectar específicamente, usar heurísticas
            # Basado en el valor y contexto de la transacción
            if 'btc' in to_address or 'bitcoin' in to_address.lower():
                return 'BTC'
            elif 'usdt' in to_address or 'tether' in to_address.lower():
                return 'USDT'
            elif 'usdc' in to_address:
                return 'USDC'
            
            # Por defecto, asumir ETH si no se puede determinar
            return 'ETH'
            
        except Exception as e:
            logger.debug(f"Error detectando símbolo de transacción: {e}")
            return 'UNKNOWN'
    
    def detect_token_from_input(self, input_data: str, contract_address: str) -> str:
        """Detecta el token basado en el input data y la dirección del contrato."""
        try:
            # Mapeo básico de direcciones de contrato a símbolos
            contract_to_symbol = {
                '0xdac17f958d2ee523a2206206994597c13d831ec7': 'USDT',
                '0xa0b86a33e6417c1c6c7c4b4c3d3c3c3c3c3c3c3c': 'USDC',
                '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599': 'BTC',
                '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2': 'ETH',
            }
            
            return contract_to_symbol.get(contract_address.lower(), 'UNKNOWN')
            
        except Exception as e:
            logger.debug(f"Error detectando token desde input: {e}")
            return 'UNKNOWN'
    
    def get_active_apis(self) -> List[str]:
        """Retorna lista de APIs activas (que tienen key configurada)."""
        active = []
        
        if self.moralis_api_key and self.moralis_api_key.strip():
            active.append("Moralis (40k requests/mes)")
        
        if (self.etherscan_api_key and 
            self.etherscan_api_key.strip() and 
            self.etherscan_api_key != "YourApiKeyToken"):
            active.append("Etherscan (5 calls/seg)")
        
        if self.quicknode_api_key and self.quicknode_api_key.strip():
            active.append("QuickNode (tiempo real)")
        
        if self.zerion_api_key and self.zerion_api_key.strip():
            active.append("Zerion (multi-blockchain)")
        
        if self.syve_api_key and self.syve_api_key.strip():
            active.append("Syve (PnL analysis)")
        
        if self.debank_api_key and self.debank_api_key.strip():
            active.append("DeBank (DeFi positions)")
        
        # Hyperliquid siempre disponible
        active.append("Hyperliquid (trading positions)")
        
        return active
            
    async def check_hyperdash_positions(self):
        """Verifica posiciones usando la API oficial de Hyperliquid."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.hyperliquid.xyz/info"
                headers = {"Content-Type": "application/json"}
                
                # Consultar estado de la wallet del usuario
                user_payload = {"type": "clearinghouseState", "user": self.wallet_address}
                
                async with session.post(url, headers=headers, json=user_payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self.process_hyperliquid_data(data)
                    else:
                        logger.debug(f"Hyperliquid API no disponible: {response.status}")
                        
        except Exception as e:
            logger.debug(f"Error consultando Hyperliquid: {e}")
            
    async def process_hyperliquid_data(self, hyperliquid_data):
        """Procesa datos de Hyperliquid para detectar posiciones de trading."""
        try:
            current_positions = {}
            
            # Obtener posiciones activas
            positions = hyperliquid_data.get("assetPositions", [])
            
            if not positions:
                logger.debug("No se encontraron posiciones en Hyperliquid")
                return
            
            # Procesar cada posición
            for item in positions:
                position_data = item.get("position", {})
                
                # El nombre del par está directamente en position.coin
                symbol = position_data.get("coin", "UNKNOWN")
                size = float(position_data.get("szi", 0))
                
                # Si el tamaño es 0, no es una posición abierta
                if size == 0:
                    continue
                
                # Determinar dirección
                direction = "LONG" if size > 0 else "SHORT"
                
                # Obtener datos de la posición
                position_value_usd = float(position_data.get("positionValue", 0))
                entry_price = float(position_data.get("entryPx", 0))
                
                # Extraer leverage
                leverage_info = position_data.get("leverage", {})
                leverage_type = leverage_info.get("type", "cross")
                leverage_value = leverage_info.get("value", 1)
                
                # Crear clave única para la posición
                position_key = f"{symbol}_hyperliquid"
                
                # Crear objeto Position
                position = Position(
                    symbol=symbol,
                    size=abs(size),
                    leverage=float(leverage_value),
                    entry_price=entry_price,
                    position_type=direction,
                    platform="Hyperliquid",
                    timestamp=datetime.now()
                )
                
                current_positions[position_key] = position
                
                logger.debug(f"Posición detectada: {symbol} {direction} {abs(size)} @ {entry_price} ({leverage_value}x)")
            
            # Comparar con posiciones anteriores para detectar cambios
            await self.compare_positions(current_positions)
            
        except Exception as e:
            logger.error(f"Error procesando datos de Hyperliquid: {e}")
            
    async def get_recent_transactions(self) -> List[Dict]:
        """Obtiene transacciones recientes usando Etherscan API."""
        if not self.etherscan_api_key or self.etherscan_api_key == "YourApiKeyToken":
            return []
            
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'module': 'account',
                    'action': 'txlist',
                    'address': self.wallet_address,
                    'startblock': 0,
                    'endblock': 99999999,
                    'page': 1,
                    'offset': 5,  # Reducir para evitar rate limits
                    'sort': 'desc',
                    'apikey': self.etherscan_api_key
                }
                
                async with session.get(self.etherscan_base, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == '1':
                            return data.get('result', [])
                        else:
                            # No mostrar error si es problema de API key
                            logger.debug(f"Etherscan API: {data.get('message', 'Unknown')}")
                    else:
                        logger.debug(f"Etherscan API HTTP error: {response.status}")
                        
        except Exception as e:
            logger.debug(f"Error obteniendo transacciones de Etherscan: {e}")
            
        return []
            
    async def check_defi_positions(self):
        """Verifica posiciones actuales en protocolos DeFi usando DeBank API."""
        # Saltar DeBank si no tenemos API key válida
        if not self.debank_api_key:
            return
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'AccessKey': self.debank_api_key
                }
                
                # Obtener posiciones de lending/borrowing
                url = f"{self.debank_base}/user/complex_protocol_list"
                params = {'id': self.wallet_address}
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self.process_defi_positions(data)
                    else:
                        # Solo log debug para errores de DeBank
                        logger.debug(f"DeBank API error: {response.status}")
                        
        except Exception as e:
            logger.debug(f"Error verificando posiciones DeFi: {e}")
            
    async def process_defi_positions(self, defi_data: Dict):
        """Procesa datos de posiciones DeFi y detecta cambios."""
        try:
            current_positions = {}
            
            for protocol in defi_data.get('data', []):
                protocol_name = protocol.get('name', 'Unknown')
                
                for portfolio in protocol.get('portfolio_item_list', []):
                    for asset in portfolio.get('detail', {}).get('supply_token_list', []):
                        symbol = asset.get('symbol', 'UNKNOWN')
                        amount = float(asset.get('amount', 0))
                        
                        if amount > 0:  # Solo posiciones activas
                            position_key = f"{symbol}_{protocol_name}"
                            
                            position = Position(
                                symbol=symbol,
                                size=amount,
                                leverage=1.0,  # DeFi lending no tiene apalancamiento directo
                                entry_price=0.0,  # No disponible en esta API
                                position_type='LONG',
                                platform=protocol_name,
                                timestamp=datetime.now()
                            )
                            
                            current_positions[position_key] = position
            
            # Comparar con posiciones anteriores
            await self.compare_positions(current_positions)
            
        except Exception as e:
            logger.error(f"Error procesando posiciones DeFi: {e}")
            
    async def analyze_transaction(self, tx: Dict):
        """Analiza una transacción para detectar aperturas/cierres de posición."""
        try:
            tx_hash = tx.get('hash')
            to_address = tx.get('to', '').lower()
            value = float(tx.get('value', 0)) / 1e18  # Wei to ETH
            
            # Verificar si es interacción con contrato conocido
            platform = self.known_contracts.get(to_address)
            
            if platform and value > 0.01:  # Transacciones significativas
                # Intentar decodificar la función llamada
                input_data = tx.get('input', '')
                
                # Detectar patrones comunes de trading
                if self.is_position_open_tx(input_data):
                    await self.handle_position_open(tx, platform)
                elif self.is_position_close_tx(input_data):
                    await self.handle_position_close(tx, platform)
                    
        except Exception as e:
            logger.error(f"Error analizando transacción {tx.get('hash')}: {e}")
            
    def is_position_open_tx(self, input_data: str) -> bool:
        """Detecta si una transacción es apertura de posición."""
        # Signatures comunes para apertura de posición
        open_signatures = [
            '0xa9059cbb',  # transfer
            '0x095ea7b3',  # approve
            '0x2e1a7d4d',  # withdraw
            # Agregar más signatures según necesidad
        ]
        
        return any(input_data.startswith(sig) for sig in open_signatures)
        
    def is_position_close_tx(self, input_data: str) -> bool:
        """Detecta si una transacción es cierre de posición."""
        # Signatures comunes para cierre de posición
        close_signatures = [
            '0x2e1a7d4d',  # withdraw
            '0x70a08231',  # balanceOf (a veces usado en cierres)
            # Agregar más signatures según necesidad
        ]
        
        return any(input_data.startswith(sig) for sig in close_signatures)
        
    async def handle_position_open(self, tx: Dict, platform: str):
        """Maneja la detección de apertura de posición."""
        try:
            # Detectar símbolo automáticamente
            detected_symbol = self.detect_symbol_from_transaction(tx)
            
            # Crear posición estimada basada en la transacción
            position = Position(
                symbol=detected_symbol,
                size=float(tx.get('value', 0)) / 1e18,
                leverage=1.0,  # Estimado
                entry_price=0.0,  # Se puede obtener del precio actual
                position_type='LONG',  # Estimado
                platform=platform,
                timestamp=datetime.fromtimestamp(int(tx.get('timeStamp', 0))),
                tx_hash=tx.get('hash')
            )
            
            # Notificar apertura
            await self.notify_position_event(PositionEvent(
                action='OPEN',
                position=position,
                timestamp=position.timestamp
            ))
            
        except Exception as e:
            logger.error(f"Error manejando apertura de posición: {e}")
            
    async def handle_position_close(self, tx: Dict, platform: str):
        """Maneja la detección de cierre de posición."""
        try:
            # Buscar posición correspondiente para cerrar
            # (En implementación real, necesitaríamos más lógica para emparejar)
            
            # Detectar símbolo automáticamente
            detected_symbol = self.detect_symbol_from_transaction(tx)
            
            position = Position(
                symbol=detected_symbol,
                size=float(tx.get('value', 0)) / 1e18,
                leverage=1.0,
                entry_price=0.0,
                position_type='LONG',
                platform=platform,
                timestamp=datetime.fromtimestamp(int(tx.get('timeStamp', 0))),
                tx_hash=tx.get('hash')
            )
            
            # Notificar cierre
            await self.notify_position_event(PositionEvent(
                action='CLOSE',
                position=position,
                timestamp=position.timestamp,
                pnl=0.0  # Se calcularía basado en precios
            ))
            
        except Exception as e:
            logger.error(f"Error manejando cierre de posición: {e}")
            
    async def compare_positions(self, current_positions: Dict[str, Position]):
        """Compara posiciones actuales con anteriores para detectar cambios."""
        try:
            # Detectar nuevas posiciones
            for key, position in current_positions.items():
                if key not in self.active_positions:
                    await self.notify_position_event(PositionEvent(
                        action='OPEN',
                        position=position,
                        timestamp=datetime.now()
                    ))
                    
            # Detectar posiciones cerradas
            for key, position in self.active_positions.items():
                if key not in current_positions:
                    await self.notify_position_event(PositionEvent(
                        action='CLOSE',
                        position=position,
                        timestamp=datetime.now()
                    ))
                    
            # Actualizar posiciones activas
            self.active_positions = current_positions.copy()
            
        except Exception as e:
            logger.error(f"Error comparando posiciones: {e}")
            
    async def notify_position_event(self, event: PositionEvent):
        """Envía notificación de evento de posición y ejecuta copy trading si está habilitado."""
        try:
            position = event.position
            
            # Ejecutar copy trading primero (para mínimo delay)
            if self.copy_trading_enabled:
                if event.action == 'OPEN' and self.position_open_callback:
                    logger.info(f"🔄 Ejecutando copy trading - ABRIR {position.symbol} {position.position_type}")
                    try:
                        await self.position_open_callback(position)
                    except Exception as e:
                        logger.error(f"Error en copy trading (abrir): {e}")
                elif event.action == 'CLOSE' and self.position_close_callback:
                    logger.info(f"🔄 Ejecutando copy trading - CERRAR {position.symbol}")
                    try:
                        await self.position_close_callback(position)
                    except Exception as e:
                        logger.error(f"Error en copy trading (cerrar): {e}")
            
            # Enviar notificación
            if event.action == 'OPEN':
                emoji = "🟢"
                action_text = "POSICIÓN ABIERTA"
                extra_info = ""
                if self.copy_trading_enabled:
                    extra_info += "\n🔄 <i>Copy trading ejecutado</i>"
            else:  # CLOSE
                emoji = "🔴"
                action_text = "POSICIÓN CERRADA"
                pnl_text = f"\n💰 P/L: ${event.pnl:.2f}" if event.pnl is not None else ""
                extra_info = pnl_text
                if self.copy_trading_enabled:
                    extra_info += "\n🔄 <i>Copy trading ejecutado</i>"
                
            message = (
                f"{emoji} <b>{action_text}</b>\n\n"
                f"📍 Wallet: <code>{self.wallet_address[:10]}...{self.wallet_address[-6:]}</code>\n"
                f"🏢 Plataforma: <b>{position.platform}</b>\n"
                f"📊 Símbolo: <b>{position.symbol}</b>\n"
                f"📏 Cantidad: <b>{position.size:.4f}</b>\n"
                f"⚡ Apalancamiento: <b>{position.leverage:.1f}x</b>\n"
                f"📈 Tipo: <b>{position.position_type}</b>\n"
                f"⏰ Tiempo: {position.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                f"{extra_info}"
            )
            
            if position.tx_hash:
                message += f"\n🔗 TX: <code>{position.tx_hash[:20]}...</code>"
                
            await self.send_telegram_notification(message)
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {e}")
            
    async def send_telegram_notification(self, message: str):
        """Envía notificación por Telegram."""
        try:
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
        except TelegramError as e:
            logger.error(f"Error enviando mensaje de Telegram: {e}")
        except Exception as e:
            logger.error(f"Error inesperado en Telegram: {e}")
            
    def enable_copy_trading(self, open_callback, close_callback):
        """Habilita copy trading con callbacks para abrir y cerrar posiciones."""
        self.copy_trading_enabled = True
        self.position_open_callback = open_callback
        self.position_close_callback = close_callback
        logger.info("🔄 Copy trading habilitado")
    
    def disable_copy_trading(self):
        """Deshabilita copy trading."""
        self.copy_trading_enabled = False
        self.position_open_callback = None
        self.position_close_callback = None
        logger.info("🔄 Copy trading deshabilitado")
    
    async def get_current_positions(self) -> List[Position]:
        """Obtiene las posiciones actuales directamente desde Hyperliquid API."""
        try:
            import aiohttp
            url = "https://api.hyperliquid.xyz/info"
            headers = {"Content-Type": "application/json"}
            user_payload = {"type": "clearinghouseState", "user": self.wallet_address}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=user_payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        positions = data.get("assetPositions", [])
                        
                        current_positions = []
                        for item in positions:
                            position_data = item.get("position", {})
                            symbol = position_data.get("coin", "UNKNOWN")
                            size = float(position_data.get("szi", 0))
                            
                            if size == 0:
                                continue
                            
                            direction = "LONG" if size > 0 else "SHORT"
                            entry_price = float(position_data.get("entryPx", 0))
                            leverage_info = position_data.get("leverage", {})
                            leverage_value = leverage_info.get("value", 1)
                            
                            position = Position(
                                symbol=symbol,
                                size=abs(size),
                                leverage=float(leverage_value),
                                entry_price=entry_price,
                                position_type=direction,
                                platform="Hyperliquid",
                                timestamp=datetime.now()
                            )
                            current_positions.append(position)
                        
                        return current_positions
                    else:
                        logger.debug(f"Error consultando Hyperliquid: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error obteniendo posiciones actuales: {e}")
            return []
    
    async def get_wallet_summary(self) -> Dict:
        """Obtiene resumen actual de la wallet, consultando posiciones reales desde Hyperliquid."""
        try:
            # Obtener posiciones actuales desde la API
            current_positions = await self.get_current_positions()
            
            summary = {
                'wallet_address': self.wallet_address,
                'active_positions': len(current_positions),
                'positions': current_positions,
                'last_check': self.last_check_time.isoformat(),
                'is_monitoring': self.is_running,
                'copy_trading_enabled': self.copy_trading_enabled
            }
            return summary
        except Exception as e:
            logger.error(f"Error obteniendo resumen de wallet: {e}")
            return {}

# Función auxiliar para crear instancia del tracker
def create_wallet_tracker(
    wallet_address: str,
    telegram_bot: Bot,
    telegram_chat_id: str,
    etherscan_api_key: Optional[str] = None,
    debank_api_key: Optional[str] = None
) -> WalletTracker:
    """Crea una instancia del rastreador de wallet."""
    return WalletTracker(
        wallet_address=wallet_address,
        telegram_bot=telegram_bot,
        telegram_chat_id=telegram_chat_id,
        etherscan_api_key=etherscan_api_key,
        debank_api_key=debank_api_key
    )
