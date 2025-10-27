# 🚂 Deploy Bot SMC en Railway - SIMPLE

## 📦 Archivos necesarios (ya creados)

- ✅ `Dockerfile` - Railway lo detecta automáticamente
- ✅ `requirements.txt` - Dependencias de Python
- ✅ `.dockerignore` - Archivos a ignorar

## 🚀 Pasos para deployar

### 1. Subir a GitHub

```bash
cd "d:\Cosas Nahuel\BotTrading"
git add .
git commit -m "Deploy bot SMC"
git push
```

### 2. Crear proyecto en Railway

1. Ir a https://railway.app
2. Click **"New Project"**
3. Seleccionar **"Deploy from GitHub repo"**
4. Elegir tu repositorio
5. Railway detecta el Dockerfile automáticamente ✨

### 3. Configurar Variables de Entorno

En Railway → **Variables** → Agregar:

```
BITUNIX_API_KEY=tu_api_key
BITUNIX_API_SECRET=tu_secret
TELEGRAM_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_chat_id
```

### 4. Configurar Root Directory (si es necesario)

Si tu bot está en `ETH/`:

Railway → **Settings** → **Root Directory** → `ETH`

### 5. Deploy

Railway hace todo automáticamente:
- Detecta el Dockerfile
- Construye la imagen
- Instala dependencias
- Ejecuta `python bot_smc_BTCETH.py`

## 🐛 Debug

Ver logs en Railway → **Deployments** → **View Logs**

## 💰 Costo

~$5-10/mes

## ✅ Listo!

Eso es todo. Railway hace el resto.
