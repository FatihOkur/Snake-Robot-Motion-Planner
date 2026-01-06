import xacro
import os
import re

# --- AYARLAR ---
# Dosya yolları (Windows/Linux uyumlu olması için / kullanıyoruz)
INPUT_FILE = "MovementAlgoSimulation/snakerobotmodel/yilansimu_description/urdf/yilansimu.xacro"
OUTPUT_FILE = "MovementAlgoSimulation/snakerobotmodel/yilansimu_description/urdf/yilansimu.urdf"

# Paket adını buraya yazın (xacro içinde $(find yilansimu_description) diye geçiyorsa)
PACKAGE_NAME = "yilansimu_description"

def convert_xacro_to_urdf():
    # Çalışma dizinini al
    cwd = os.getcwd()
    
    # Tam dosya yolunu oluştur
    abs_input_path = os.path.abspath(INPUT_FILE)
    
    # Paket klasörünün tam yolunu bulmaya çalış (urdf klasörünün bir üstü)
    # Varsayım: urdf dosyası .../paket_adi/urdf/dosya.xacro dizininde
    package_path = os.path.dirname(os.path.dirname(abs_input_path))
    
    # Windows ters slash sorununu düzelt (path'leri / formatına çevir)
    package_path = package_path.replace("\\", "/")
    
    if not os.path.exists(INPUT_FILE):
        print(f"HATA: Girdi dosyası bulunamadı: {INPUT_FILE}")
        print(f"Aranan Tam Yol: {abs_input_path}")
        return

    print(f"İşleniyor: {INPUT_FILE}...")
    print(f"Sanal Paket Yolu: {package_path}")

    try:
        # 1. Dosyayı metin olarak oku
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            xacro_content = f.read()

        # 2. $(find paket_adi) komutlarını gerçek dosya yoluyla değiştir (Manuel Patch)
        # ROS kurulu olmadığında xacro bu komutu işleyemez, biz elle yapıyoruz.
        find_pattern = f"\\$\\(find {PACKAGE_NAME}\\)"
        if re.search(find_pattern, xacro_content):
            print(f"Bilgi: '$(find {PACKAGE_NAME})' komutları yerel yollarla değiştiriliyor...")
            xacro_content = re.sub(find_pattern, package_path, xacro_content)

        # 3. Xacro motorunu çalıştır (process_doc kullanarak string üzerinden git)
        doc = xacro.parse(xacro_content)
        xacro.process_doc(doc)
        xml_string = doc.toxml()

        # 4. URDF olarak kaydet
        # Klasör yoksa oluştur
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
            f.write(xml_string)
        
        print(f"BAŞARILI! Dosya oluşturuldu: {OUTPUT_FILE}")

    except Exception as e:
        print("\n--- BİR HATA OLUŞTU ---")
        print(str(e))
        print("-----------------------")
        print("İpucu: Eğer hala path hatası alıyorsanız, xacro dosyanızda")
        print("başka paketlere referanslar ($(find baska_paket)) olabilir.")

if __name__ == "__main__":
    convert_xacro_to_urdf()