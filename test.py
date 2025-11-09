import asyncio
from bleak import BleakClient

ESP32_BLE_ADDRESS = "34:94:54:F0:45:7A"  # bez dvojbodky na konci
CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

async def test_ble():
    async with BleakClient(ESP32_BLE_ADDRESS) as client:
        if client.is_connected:
            print("✅ Pripojený k ESP32")
            # pošli testovací príkaz
            await client.write_gatt_char(CHARACTERISTIC_UUID, "on".encode())
            print("LED by sa mala rozsvietiť")
        else:
            print("❌ Nepodarilo sa pripojiť")

asyncio.run(test_ble())
