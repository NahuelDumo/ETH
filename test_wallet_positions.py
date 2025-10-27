import aiohttp
import asyncio
import json

# Ya no se necesita clave de API
# ZAPPER_API_KEY = "..." 

async def get_hyperliquid_positions(wallet_address):
    """
    Obtiene las posiciones de un usuario desde la API de Hyperliquid.
    Los nombres de los pares vienen directamente en position.coin.
    """
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        # Obtener el estado de la wallet del usuario
        try:
            print("🛰️  Consultando estado de la wallet en Hyperliquid...")
            user_payload = {"type": "clearinghouseState", "user": wallet_address}
            async with session.post(url, headers=headers, json=user_payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    positions = data.get("assetPositions", [])
                    margin_summary = data.get("marginSummary", {})
                    return positions, margin_summary
                else:
                    print(f"Error HTTP {resp.status} al obtener estado de la wallet.")
                    print(await resp.text())
                    return None, None
        except Exception as e:
            print(f"❌ Excepción al obtener estado de la wallet: {e}")
            return None, None

async def main():
    wallet = "0xc2a30212a8ddac9e123944d6e29faddce994e5f2"
    print(f"🔍 Analizando posiciones de Hyperliquid para: {wallet}")
        
    positions, margin_summary = await get_hyperliquid_positions(wallet)

    if positions is None:
        print("⚠️ Error al contactar la API de Hyperliquid.")
        return

    print("\n--- Resumen de la Cuenta ---")
    if margin_summary:
        total_margin = float(margin_summary.get("totalMargin", 0))
        total_pnl = float(margin_summary.get("totalUnrealizedPnl", 0))
        print(f"💰 Valor Total de la Cuenta (Margen): ${total_margin:,.2f}")
        print(f"📈 PnL Total No Realizado: ${total_pnl:,.2f}")
    
    print("\n--- Posiciones Abiertas ---")
    
    found_positions = False
    if not positions:
         print("No se encontraron 'assetPositions' (la lista está vacía).")
         return

    for item in positions:
        position_data = item.get("position", {})
        
        # El nombre del par está directamente en position.coin
        asset = position_data.get("coin", "N/A")
        
        size = float(position_data.get("szi", 0))
        
        # Si 'szi' (tamaño) es 0, no es una posición abierta. La saltamos.
        if size == 0:
            continue
            
        found_positions = True
        direction = "LONG" if size > 0 else "SHORT"
        
        # Corregir: el campo es "positionValue" no "value"
        position_value_usd = float(position_data.get("positionValue", 0))
        unrealized_pnl = float(position_data.get("unrealizedPnl", 0))
        entry_price = float(position_data.get("entryPx", 0))
        
        # Extraer leverage correctamente
        leverage_info = position_data.get("leverage", {})
        leverage_type = leverage_info.get("type", "cross")
        leverage_value = leverage_info.get("value", 0)
        
        # Formatear el leverage para mostrar "Cross" o "9x"
        if leverage_type == "cross":
            leverage_str = f"Cross ({leverage_value}x)" if leverage_value > 0 else "Cross"
        else:  # isolated
            leverage_str = f"{leverage_value}x"
        
        liq_price = float(position_data.get("liquidationPx", 0))
        
        print(f"🌐 {asset} ({leverage_str})")
        print(f"   - Dirección: {direction}")
        print(f"   - Valor de Posición: ${position_value_usd:,.2f}")
        print(f"   - PnL No Realizado: ${unrealized_pnl:,.2f}")
        print(f"   - Precio de Entrada: ${entry_price:,.2f}")
        print(f"   - Precio de Liq.: ${liq_price:,.2f}")
        print()
        
    if not found_positions:
        print("No se encontraron posiciones abiertas (szi != 0) en los activos devueltos.")

if __name__ == "__main__":
    asyncio.run(main())