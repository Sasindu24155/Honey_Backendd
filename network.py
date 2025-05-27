import os
import sys
import ctypes
import time
import threading

try:
    import pydivert
except ImportError:
    print("Please install pydivert: pip install pydivert")
    sys.exit(1)

try:
    import mysql.connector
except ImportError:
    print("Please install a MySQL driver: pip install mysql-connector-python")
    sys.exit(1)

# ——— Database connection info — adjust these ———
DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = ""
DB_NAME = "netcontrol"
DB_TABLE= "speed_limit"
DB_ID   = 1
# —————————————————————————————————————

DRIVER_DLL = os.path.join(os.path.dirname(__file__), "WinDivert.dll")

# ——— Shared state — token‑buckets in BYTES ———
download_b_per_sec = 0.0
upload_b_per_sec   = 0.0
download_tokens    = 0.0
upload_tokens      = 0.0
last_refill        = time.time()

lock = threading.Lock()

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def elevate():
    params = " ".join(f'"{arg}"' for arg in sys.argv)
    ctypes.windll.shell32.ShellExecuteW(None, "runas",
                                        sys.executable,
                                        params, None, 1)
    sys.exit(0)

def fetch_limits_loop():
    global download_b_per_sec, upload_b_per_sec, download_tokens, upload_tokens
    while True:
        try:
            conn = mysql.connector.connect(
                host=DB_HOST, user=DB_USER,
                password=DB_PASS, database=DB_NAME
            )
            cur = conn.cursor()
            cur.execute(
                f"SELECT download_kbps, upload_kbps "
                f"FROM {DB_TABLE} WHERE id = %s",
                (DB_ID,)
            )
            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                d_kbps, u_kbps = row
                # convert kbps→bps→bytes/sec
                d_bps = d_kbps * 1000
                u_bps = u_kbps * 1000
                with lock:
                    download_b_per_sec = d_bps / 8.0
                    upload_b_per_sec   = u_bps / 8.0
                    # clamp any tokens above the new rate
                    download_tokens = min(download_tokens, download_b_per_sec)
                    upload_tokens   = min(upload_tokens,   upload_b_per_sec)

                # print new limits
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"New caps → download: {d_kbps} kbps ({d_kbps/1000:.2f} Mbps), "
                    f"upload: {u_kbps} kbps ({u_kbps/1000:.2f} Mbps)"
                )
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] DB fetch error:", e)

        time.sleep(5)

def refill_buckets():
    global last_refill, download_tokens, upload_tokens
    while True:
        now = time.time()
        with lock:
            elapsed = now - last_refill
            download_tokens = min(
                download_b_per_sec,
                download_tokens + elapsed * download_b_per_sec
            )
            upload_tokens = min(
                upload_b_per_sec,
                upload_tokens   + elapsed * upload_b_per_sec
            )
            last_refill = now
        time.sleep(0.01)

def shape_traffic():
    global download_tokens, upload_tokens
    try:
        with pydivert.WinDivert("true") as w:
            for packet in w:
                pkt_len = len(packet.raw)
                is_out = packet.is_outbound
                while True:
                    with lock:
                        if is_out:
                            if upload_tokens >= pkt_len:
                                upload_tokens -= pkt_len
                                break
                        else:
                            if download_tokens >= pkt_len:
                                download_tokens -= pkt_len
                                break
                    time.sleep(0.005)
                w.send(packet)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    if not is_admin():
        elevate()

    if not os.path.exists(DRIVER_DLL):
        print(f"Error: '{DRIVER_DLL}' not found.")
        print("Download WinDivert from https://reqrypt.org/windivert.html and place WinDivert.dll here.")
        sys.exit(1)

    print("Starting dynamic network shaper.")
    print("  → Limits fetched every 5 s from database.")
    print("  → Ctrl+C to stop and remove caps.\n")

    threading.Thread(target=fetch_limits_loop, daemon=True).start()
    threading.Thread(target=refill_buckets,   daemon=True).start()
    shape_traffic()

    print("\nExited. Caps removed.")
