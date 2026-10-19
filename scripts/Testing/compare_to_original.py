import gguf
import config

official_path = config.ORIGINALMODEL # Pfad zum funktionierenden Original
custom_guff_path = config.OUTPUT_GGUF_PATH

def dump_meta(path):
    print(f"\n--- METADATEN FÜR: {path} ---")
    reader = gguf.GGUFReader(path)
    for key, field in reader.fields.items():
        # Wir zeigen alle Konfigurationen, Layer, Typen und Feature-Flags
        print(f"{key}: type={field.types}, val={field.parts[-1] if hasattr(field, 'parts') else field}")

dump_meta(official_path)
dump_meta(custom_guff_path)