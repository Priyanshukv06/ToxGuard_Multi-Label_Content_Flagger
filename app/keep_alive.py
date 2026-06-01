"""
Health-check ping
"""

import os
import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_HOURS = 12

async def keep_alive_loop():
    service_url = os.getenv("SERVICE_URL", "")

    if not service_url:
        hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "")
        if hostname:
            service_url = f"https://{hostname}"

    if not service_url:
        logger.info("ℹ️ No SERVICE_URL. Keep-alive disabled.")
        return

    try:
        interval_hours = float(os.getenv("KEEP_ALIVE_INTERVAL_HOURS", str(DEFAULT_INTERVAL_HOURS)))
    except ValueError:
        interval_hours = DEFAULT_INTERVAL_HOURS

    if interval_hours <= 0:
        return

    interval_seconds = int(interval_hours * 3600)
    ping_url = f"{service_url}/health"

    logger.info(f"🏓 Keep-alive scheduled: {ping_url} every {interval_hours}h")

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            async with httpx.AsyncClient() as client:
                await client.get(ping_url, timeout=60)
        except Exception as e:
            logger.warning(f"🏓 Keep-alive ping failed: {e}")
