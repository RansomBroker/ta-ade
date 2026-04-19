import os
import re
import csv
from bs4 import BeautifulSoup

# ===============================
# KONFIGURASI
# ===============================
RAW_DATA_DIR = "raw-data"
OUTPUT_FILE = "data-okupansi-hotel.csv"

HEADERS = ["No", "Waktu", "Reg No", "Nama Tamu", "Room No",
           "Room Rate", "Diskon", "Extra Bed", "Deposit",
           "Keterangan", "FO Clerk"]


def parse_htm_file(file_path):
    """Parse satu file HTM dan kembalikan list of rows (dict)."""
    rows = []

    with open(file_path, "r", encoding="windows-1252", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Cari semua TR
    all_trs = soup.find_all("tr")

    for tr in all_trs:
        # Cari semua FONT SIZE=1 di dalam TR ini
        fonts = tr.find_all("font", attrs={"size": "1"})
        if not fonts:
            continue

        # Ambil teks dari setiap FONT
        texts = [font.get_text(strip=True) for font in fonts]

        # Filter: baris data dimulai dengan nomor urut (misal "1.", "25.", "100.")
        if not texts or not re.match(r'^\d+\.$', texts[0]):
            continue

        # Harus ada tepat 11 kolom (No, Waktu, Reg No, Nama Tamu, Room No,
        # Room Rate, Diskon, Extra Bed, Deposit, Keterangan, FO Clerk)
        if len(texts) != 11:
            continue

        row = {
            "No": texts[0].rstrip('.'),
            "Waktu": texts[1],
            "Reg No": texts[2],
            "Nama Tamu": texts[3],
            "Room No": texts[4],
            "Room Rate": texts[5],
            "Diskon": texts[6],
            "Extra Bed": texts[7],
            "Deposit": texts[8],
            "Keterangan": texts[9],
            "FO Clerk": texts[10],
        }
        rows.append(row)

    return rows


def get_sort_key(filename):
    """Ambil (tahun, bulan) dari nama file seperti '2021_1.htm'."""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    return (int(parts[0]), int(parts[1]))


def main():
    # Kumpulkan semua file HTM, urutkan berdasarkan tahun_bulan
    htm_files = sorted(
        [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".htm")],
        key=get_sort_key
    )

    print(f"Ditemukan {len(htm_files)} file HTM di '{RAW_DATA_DIR}'")

    all_rows = []
    global_no = 1

    for htm_file in htm_files:
        file_path = os.path.join(RAW_DATA_DIR, htm_file)
        print(f"  Memproses: {htm_file} ... ", end="")

        rows = parse_htm_file(file_path)
        print(f"{len(rows)} baris data")

        # Re-number secara global dan tambahkan ke list
        for row in rows:
            row["No"] = str(global_no)
            global_no += 1
            all_rows.append(row)

    # Tulis ke CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSelesai! Total {len(all_rows)} baris ditulis ke '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
